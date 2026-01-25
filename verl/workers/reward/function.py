# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

from sympy import group

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


ScoreFunction = Callable[[str, str], RewardScore]




@dataclass
class FunctionRewardManager:
    config: RewardConfig
    tokenizer: PreTrainedTokenizer

    def __post_init__(self):
        """Load score function."""
        if ":" not in self.config.score_function:
            file_path = self.config.score_function
            function_name = "main"
        else:
            file_path, function_name = self.config.score_function.split(":", maxsplit=1)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Score function file {file_path} not found.")

        spec = importlib.util.spec_from_file_location("custom_score_fn", file_path)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_score_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load score function: {e}")

        if not hasattr(module, function_name):
            raise AttributeError(f"Module {module} does not have function {function_name}.")

        score_fn: ScoreFunction = getattr(module, function_name)
        print(f"Using score function `{function_name}` from `{file_path}`.")
        self.score_fn = partial(score_fn, **self.config.score_function_kwargs)
        
        
    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        
        n = 8
        
        group = []
        lcr1_length_w = 1.0
        lcr1_compress_w = 1.0
        lcr1_think_start_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
        lcr1_think_end_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]
            lcr1_valid_len = int(response_mask.sum())
            lcr1_valid_ids = response_ids[:lcr1_valid_len]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data_item.non_tensor_batch["ground_truth"]
            score = self.score_fn(response_str, ground_truth)
            if self.config.length_reward == "LP1":
                if len(response_str) >= 6144:
                   score["overall"] -= min(0.5, (len(response_str) - 6144) / 4096)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)
            elif self.config.length_reward == "LP2": 
                lp2_weight = 1.0
                correct01 = 1 if float(score["overall"]) > 0.5 else 0 
                group.append((i, valid_response_length, score, correct01)) 
                if len(group) == n: 
                    self._flush_group_lp2(group, reward_tensor, reward_metrics, lp2_weight)
                continue
            elif self.config.length_reward == "ShorterBetter":
                sb_alpha = 1.0
                sb_beta  = float(1e-3)
                correct01 = 1 if float(score["overall"]) > 0.5 else 0
                group.append((i, valid_response_length, score, correct01))
                if len(group) == n:
                    self._flush_group_shorterbetter(
                        group, reward_tensor, reward_metrics, alpha=sb_alpha, beta=sb_beta
                    )
                continue
            elif self.config.length_reward == "LC-R1":
                self._lcr1_push(
                    group=group,
                    idx=i,
                    valid_ids=lcr1_valid_ids,
                    valid_len=lcr1_valid_len,
                    score=score,
                    ground_truth=ground_truth,
                    reward_tensor=reward_tensor,
                    reward_metrics=reward_metrics,
                    n=n,
                    length_weight=lcr1_length_w,
                    compress_weight=lcr1_compress_w,
                    think_start_ids=lcr1_think_start_ids,
                    think_end_ids=lcr1_think_end_ids,
                )
                # LC-R1 由 flush 写入 reward_tensor/metrics，这里不要重复写
                continue
            else:                       
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                for key, value in score.items():
                    reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
    
        
    def _flush_group_lp2(self, group, reward_tensor, reward_metrics, lp2_weight: float):
        if not group:
            return group
        lens = [g[1] for g in group]
        min_len = min(lens)
        max_len = max(lens)
        if max_len == min_len:
            len_rewards = [0.0] * len(group)
        else:
            denom = float(max_len - min_len)
            len_rewards = []
            for (_, l, _, correct01) in group:
                lam = 0.5 - (float(l - min_len) / denom)  # in [-0.5, 0.5]
                if correct01 == 1:
                    lr = lam
                else:
                    lr = min(0.0, lam) 
                len_rewards.append(lr)
        for (idx, valid_len, score, _), lr in zip(group, len_rewards): 
            score["overall"] = float(score["overall"]) + lp2_weight * float(lr) 
            reward_tensor[idx, valid_len - 1] = score["overall"] 
            for key, value in score.items(): 
                reward_metrics[key].append(float(value))



        group.clear()
        return group
    
    def _flush_group_shorterbetter(
        self,
        group,
        reward_tensor,
        reward_metrics,
        alpha: float,
        beta: float,
    ):
        """
        group item: (idx, valid_len, score_dict, correct01)
        reward: alpha*correct01 - beta*abs(valid_len - SOL)
        SOL: min len among correct if any correct else mean len
        """
        if not group:
            return group

        lens = [int(g[1]) for g in group]
        correct_lens = [int(l) for (_, l, _, c) in group if int(c) == 1]

        if len(correct_lens) > 0:
            sol_len = min(correct_lens)
        else:
            sol_len = sum(lens) / float(len(lens))  # mean length when all wrong

        for (idx, valid_len, score, correct01) in group:
            valid_len = int(valid_len)
            correct01 = int(correct01)

            sb_reward = float(alpha) * float(correct01) - float(beta) * abs(float(valid_len) - float(sol_len))

            # 如果你想“完全按 ShorterBetter”，就覆盖 overall：
            score["overall"] = float(sb_reward)

            # 额外记录一些指标（可选，但很有用）
            score["sb_sol_len"] = float(sol_len)
            score["sb_len"] = float(valid_len)
            score["sb_correct01"] = float(correct01)

            reward_tensor[idx, valid_len - 1] = float(score["overall"])
            for k, v in score.items():
                reward_metrics[k].append(float(v))
        group.clear()
        return group
    
    def _find_subseq(self, haystack, needle):
        """Return start idx of needle in haystack, else -1. Both are Python lists[int]."""
        if not needle or len(needle) > len(haystack):
            return -1
        L = len(needle)
        for i in range(len(haystack) - L + 1):
            if haystack[i:i + L] == needle:
                return i
        return -1
    
    def _flush_group_lcr1(
        self,
        group,
        reward_tensor,
        reward_metrics,
        length_weight: float,
        compress_weight: float,
    ):
        """
        group item dict keys:
        idx, valid_len, score(dict), correct01(int),
        comp_total_len(int), r_comp(float), think_end_pos(int|None)
        """
        if not group:
            return group

        # Length reward only considers correct samples
        correct_comp_lens = [g["comp_total_len"] for g in group if g["correct01"] == 1]
        max_len_c = max(correct_comp_lens) if correct_comp_lens else None

        for g in group:
            idx = g["idx"]
            valid_len = g["valid_len"]
            score = g["score"]
            correct01 = g["correct01"]
            comp_total_len = g["comp_total_len"]
            r_comp = g["r_comp"]
            think_end_pos = g["think_end_pos"]

            # r_len = 1 - |o'| / max_{j in C} |o'_j|, else 0
            if correct01 == 1 and max_len_c is not None and max_len_c > 0:
                r_len = 1.0 - float(comp_total_len) / float(max_len_c)
            else:
                r_len = 0.0

            # 记录额外指标（可选）
            score["length_reward"] = float(r_len)
            score["compress_reward"] = float(r_comp)

            # base overall + length reward
            score["overall"] = float(score["overall"]) + float(length_weight) * float(r_len)

            # terminal reward
            reward_tensor[idx, valid_len - 1] = float(score["overall"])

            # compress reward ONLY at </think> token position
            if think_end_pos is not None and 0 <= think_end_pos < valid_len:
                reward_tensor[idx, think_end_pos] += float(compress_weight) * float(r_comp)

            for k, v in score.items():
                reward_metrics[k].append(float(v))

        group.clear()
        return group


    def _lcr1_build_item(
        self,
        idx: int,
        valid_ids,          # 1D torch.Tensor (already truncated to valid_len)
        valid_len: int,
        score: dict,
        ground_truth,
        think_start_ids: list,
        think_end_ids: list,
    ):
        """把单条样本转成 LC-R1 group item（不写 reward，等 flush 统一写）"""
        correct01 = 1 if float(score["overall"]) > 0.5 else 0

        ids_list = valid_ids.tolist()

        # locate </think>
        end_tag_start = self._find_subseq(ids_list, think_end_ids)
        think_end_pos = end_tag_start + len(think_end_ids) - 1 if end_tag_start >= 0 else None

        # locate <think>
        start_tag_start = self._find_subseq(ids_list, think_start_ids)
        think_content_start = start_tag_start + len(think_start_ids) if start_tag_start >= 0 else 0
        think_content_end = end_tag_start if end_tag_start >= 0 else valid_len

        thinking_ids = ids_list[think_content_start:think_content_end]
        orig_think_len = len(thinking_ids)

        # conservative compress estimation: find GT token subseq inside thinking
        r_comp = 0.0
        comp_total_len = valid_len

        if correct01 == 1 and orig_think_len > 0:
            gt_ids = self.tokenizer.encode(str(ground_truth), add_special_tokens=False)
            pos = self._find_subseq(thinking_ids, gt_ids) if gt_ids else -1

            if pos >= 0:
                comp_think_len = pos + len(gt_ids)
                removed = orig_think_len - comp_think_len
                if removed > 0:
                    comp_total_len = valid_len - removed
                    r_comp = 1.0 - float(comp_think_len) / float(orig_think_len)
            else:
                # 找不到“首次正确”位置：不施加压缩奖励，避免误伤
                r_comp = 0.0

        return {
            "idx": idx,
            "valid_len": int(valid_len),
            "score": score,
            "correct01": int(correct01),
            "comp_total_len": int(comp_total_len),
            "r_comp": float(r_comp),
            "think_end_pos": think_end_pos,
        }


    def _lcr1_push(
        self,
        group: list,
        idx: int,
        valid_ids,
        valid_len: int,
        score: dict,
        ground_truth,
        reward_tensor,
        reward_metrics,
        n: int,
        length_weight: float,
        compress_weight: float,
        think_start_ids: list,
        think_end_ids: list,
    ):
        """push 一个样本进 LC-R1 group；满 n 则 flush。返回 group（原地修改）"""
        item = self._lcr1_build_item(
            idx, valid_ids, valid_len, score, ground_truth, think_start_ids, think_end_ids
        )
        group.append(item)
        if len(group) == n:
            self._flush_group_lcr1(group, reward_tensor, reward_metrics, length_weight, compress_weight)
        return group


    def _lcr1_finalize(
        self,
        group: list,
        reward_tensor,
        reward_metrics,
        length_weight: float,
        compress_weight: float,
    ):
        """循环结束 flush 残余（避免最后不足 n 的样本完全不吃 LC-R1）"""
        if group:
            self._flush_group_lcr1(group, reward_tensor, reward_metrics, length_weight, compress_weight)
        return group



