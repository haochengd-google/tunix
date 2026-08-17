# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM rollout worker process runner for the distributed GRPO demo."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pickle
import sys

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from transformers import AutoTokenizer
from tunix.experimental.worker import raiden_tpu_worker
from tunix.experimental.rollout import legacy_vllm_sampler_adapter
from tunix.experimental.rollout import vanilla_sampler_adapter
from tunix.models.gemma import model as gemma_model_lib
from tunix.models.gemma import params_safetensors as gemma_params_lib
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import rollout_worker
from tunix.generate import mappings as mappings_lib
from tunix.generate import tokenizer_adapter as tokenizer_adapter_lib
from tunix.generate import vllm_sampler
from tunix.models.qwen3 import mapping_vllm_jax
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="vLLM rollout worker process")
  parser.add_argument("--port", type=int, default=20001)
  parser.add_argument("--worker_id", type=str, default="vllm-rollout-0")
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument(
      "--model_dir", type=str, default=os.getenv("MODEL_DIR", "")
  )
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--max_prompt_length", type=int, default=512)
  parser.add_argument("--max_response_length", type=int, default=128)
  parser.add_argument("--use_lora", action="store_true")
  parser.add_argument("--lora_rank", type=int, default=16)
  parser.add_argument("--lora_alpha", type=float, default=16.0)
  parser.add_argument(
      "--sampler",
      type=str,
      default=os.getenv("SAMPLER", "legacy_vllm"),
      choices=["legacy_vllm", "vanilla_gemma"],
  )
  parser.add_argument(
      "--gemma_config",
      type=str,
      default=os.getenv("GEMMA_CONFIG", "gemma2_2b"),
      choices=["gemma_2b", "gemma2_2b"],
  )
  return parser.parse_args(argv)


def _create_rollout_mesh() -> Mesh:
  shape = (1, jax.device_count())
  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  return Mesh(devices, axis_names=("dp", "tp"))


class _GSM8KDemoEnv(base_environment.BaseTaskEnv):
  """Minimal single-step math environment for the distributed demo."""

  def __init__(
      self,
      prompt: str = "",
      gold_answer: str = "",
      group_id: str = "",
      pair_index: int = 0,
      policy_version: int = 0,
      max_steps: int = 1,
      **kwargs: Any,
  ):
    super().__init__(
        task={
            "prompts": prompt,
            "gold_answer": gold_answer,
            "policy_version": policy_version,
        },
        max_steps=max_steps,
        group_id=group_id,
        pair_index=pair_index,
        **kwargs,
    )

  def _initial_observation(self) -> dict[str, str]:
    return {"prompts": self.task.get("prompts", "")}

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    answer = str(action)
    gold_answer = str(self.task.get("gold_answer", ""))
    is_correct = bool(gold_answer) and gold_answer in answer
    return base_environment.EnvStepResult(
        observation={"answer": answer, "gold_answer": gold_answer},
        reward=1.0 if is_correct else 0.0,
        done=True,
        info={"correct": is_correct},
    )


class _GSM8KDemoEnvPool:
  """Creates one lightweight environment per rollout request."""

  def acquire_env(
      self, config: dict[str, Any] | None = None
  ) -> _GSM8KDemoEnv:
    return _GSM8KDemoEnv(**dict(config or {}))

  def release_env(self, env: _GSM8KDemoEnv) -> None:
    env.close()


class _GSM8KDemoAgent(base_agent.ConversationAgentBase):
  """Minimal agent that forwards model text as the environment action."""

  name = "gsm8k_demo_agent"

  def __init__(self):
    super().__init__(
        "Solve the math problem. Return the final numeric answer clearly."
    )

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    del kwargs
    action = agent_types.Action(action=response)
    self.trajectory.steps.append(
        agent_types.Step(
            model_response=response,
            thought="",
            action=action,
        )
    )
    self.chat_completions.append({"role": "assistant", "content": response})
    return action


class _RaidenVanillaSampler(vanilla_sampler_adapter.VanillaSamplerAdapter):
  """Native gemma sampler facade; device work runs on its TPU workers."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._tpu_workers = [raiden_tpu_worker.RaidenTpuWorker("rollout")]
    self._version = 0

  def _bound_workers(self):
    for worker in self._tpu_workers:
      worker.bind(self.sampler.transformer_state)
    return self._tpu_workers

  async def bind_weight_sync(self, **kwargs):
    del kwargs
    self._bound_workers()
    return None

  async def get_weight_sync_metadata(self, **kwargs):
    del kwargs
    return [w.work_unit_metadata() for w in self._bound_workers()]

  async def pre_weight_sync(self, sync_request=None, **kwargs):
    del sync_request, kwargs
    self._bound_workers()
    return True

  async def weight_sync(self, sync_request=None, **kwargs):
    del kwargs
    for worker in self._bound_workers():
      worker.h2d()
      if os.environ.get("VERIFY_WEIGHTS", "").lower() == "true":
        logging.info("destination checksums: %s", worker.checksums())
    self._version = getattr(sync_request, "policy_version", self._version + 1)
    return self._version

  async def post_weight_sync(self, sync_request=None, **kwargs):
    del sync_request, kwargs
    for worker in self._tpu_workers:
      logging.info("raiden metrics: %s", worker.metrics())
    return True


def _create_gemma_vanilla_worker(args, tokenizer):
  logging.info("Creating native gemma sampler on the rollout mesh...")
  mesh = _create_rollout_mesh()
  model_config = getattr(
      gemma_model_lib.ModelConfig, args.gemma_config
  )()
  with mesh:
    model = gemma_params_lib.create_model_from_safe_tensors(
        args.model_dir or args.model_id, model_config, mesh=mesh
    )
  sampler_adapter = _RaidenVanillaSampler(
      server_id=args.worker_id,
      transformer=model,
      tokenizer=tokenizer,
      cache_config=args.max_prompt_length + args.max_response_length,
  )
  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = chat_parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=False
  )
  config = rollout_worker.RolloutConfig(
      sampler_type="vanilla",
      max_prompt_length=args.max_prompt_length,
      max_tokens_to_generate=args.max_response_length,
      temperature=1.0,
      top_p=1.0,
      return_logprobs=True,
  )
  return rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
      config=config,
      sampler=sampler_adapter,
      env_pool=_GSM8KDemoEnvPool(),
      agent_factory=_GSM8KDemoAgent,
      tokenizer=rollout_tokenizer,
      chat_parser=chat_parser,
      max_concurrency=64,
  )


def _create_vllm_worker(args, tokenizer):
  logging.info("Creating vLLM mapping config...")
  mapping_config = mappings_lib.MappingConfig(
      lora_to_hf_mappings=mapping_vllm_jax.LORA_TO_HF_MAPPINGS
  )
  vllm_model = args.model_dir or args.model_id
  rollout_mesh = _create_rollout_mesh()
  max_model_len = args.max_prompt_length + args.max_response_length
  logging.info(
      "Creating vLLM config for model=%s mesh=%s tensor_parallel_size=%d "
      "max_model_len=%d...",
      vllm_model,
      rollout_mesh,
      jax.device_count(),
      max_model_len,
  )
  vllm_config = vllm_sampler.VllmConfig(
      mesh=rollout_mesh,
      tensor_parallel_size=jax.device_count(),
      data_parallel_size=1,
      return_logprobs=True,
      lora_config=(
          {
              "max_lora_rank": args.lora_rank,
              "max_loras": 1,
          }
          if args.use_lora
          else None
      ),
      mapping_config=mapping_config,
      engine_kwargs={
          "model": vllm_model,
          "max_model_len": max_model_len,
      },
  )
  sampler_adapter = legacy_vllm_sampler_adapter.LegacyVllmSamplerAdapter(
      server_id=args.worker_id,
      tokenizer=tokenizer,
      config=vllm_config,
  )
  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = chat_parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=False
  )
  logging.info("Creating RolloutWorker wrapper...")
  config = rollout_worker.RolloutConfig(
      sampler_type="legacy_vllm",
      max_prompt_length=args.max_prompt_length,
      max_tokens_to_generate=args.max_response_length,
      temperature=1.0,
      top_p=1.0,
      return_logprobs=True,
      rollout_vllm_model_version=vllm_model,
  )
  return rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
      config=config,
      sampler=sampler_adapter,
      env_pool=_GSM8KDemoEnvPool(),
      agent_factory=_GSM8KDemoAgent,
      tokenizer=rollout_tokenizer,
      chat_parser=chat_parser,
      max_concurrency=64,
  )


def main(argv: list[str], context: Any = None) -> None:
  if context and context.ipc and context.ipc.discovery:
    pass
  else:
    raise RuntimeError(
        "Require discovery API, but process context doesn't support."
    )

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [RolloutNode] %(message)s",
      force=True,
  )

  args = _parse_args(argv)
  logging.info("Parsed args: %s", args)

  if context:
    context.jax.initialize()
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)


  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  logging.info("Loading tokenizer from %s...", tokenizer_path)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

  async def grpc_server_main() -> None:
    logging.info("Creating rollout worker service (%s)...", args.sampler)
    if args.sampler == "vanilla_gemma":
      worker_service = _create_gemma_vanilla_worker(args, tokenizer)
    else:
      worker_service = _create_vllm_worker(args, tokenizer)

    logging.info("Creating rollout gRPC server...")
    server = remote_execution.GrpcRemoteExecutionServer(worker_service)
    await server.start_serving_async(args.port)
    logging.info("Serving vLLM rollout worker on port %d.", args.port)

    context.ipc.discovery.register(
        metadata=pickle.dumps({
            "service_type": "rollout",
            "service_port": args.port,
            "worker_id": args.worker_id,
        })
    )
    logging.info("Rollout worker is registered.")

    try:
      while True:
        await asyncio.sleep(1)
    except asyncio.CancelledError:
      pass
    finally:
      await server.stop_serving()

  asyncio.run(grpc_server_main())


if __name__ == "__main__":
  main(sys.argv[1:])

