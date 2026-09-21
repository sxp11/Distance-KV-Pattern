#!/usr/bin/env python3
"""Eight-rank retrieval initializer; each rank owns whole contexts and raw output."""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
from typing import Any, Callable
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

METHOD_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(METHOD_ROOT / "src"))
from distance_kv_pattern import BlockLayout, InstructMaterializer, LogProgressBar, read_manifest
from distance_kv_pattern.training.retrieval_initialization.retrieval_init import (RAW_SCHEMA_VERSION, ATTENTION_METRICS,
    aggregate_raw_records_condition_balanced, append_jsonl, read_jsonl, save_aggregate)
from distance_kv_pattern.core.randomness import set_reproducibility
from .collect_retrieval_initialization import (build_plan, branch_id,
    resolve_dtype, plan_summary)

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-name-or-path',required=True); p.add_argument('--manifest-root',type=Path); p.add_argument('--manifest-paths',nargs='+',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--value-types',nargs='+',default=('num','word')); p.add_argument('--needle-count',type=int,default=4); p.add_argument('--template-ids',nargs='+',default=('Q1','Q3'))
    p.add_argument('--coverage-round',type=int,default=0); p.add_argument('--retrieval-k',type=int,default=5); p.add_argument('--top-q',type=int,default=2); p.add_argument('--master-seed',type=int,default=20260819)
    p.add_argument('--torch-dtype',choices=('bfloat16','float16'),default='bfloat16'); p.add_argument('--attn-implementation',default='flash_attention_2'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--limit-contexts',type=int)
    return p.parse_args()

def main(
    score_branch_fn: Callable[..., dict[str, Any]],
    model_loader: Callable[..., Any] = AutoModelForCausalLM.from_pretrained,
    tokenizer_loader: Callable[..., Any] = AutoTokenizer.from_pretrained,
    layout_factory: Callable[[], BlockLayout] = BlockLayout,
    materializer_factory: Callable[..., Any] = InstructMaterializer,
):
    a=args(); set_reproducibility(a.master_seed); rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    if world>1: torch.distributed.init_process_group('nccl'); torch.cuda.set_device(rank)
    layout=layout_factory(); contexts=build_plan(a)
    if rank==0: a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/'plan.json').write_text(json.dumps({'schema_version':RAW_SCHEMA_VERSION,'master_seed':a.master_seed,'world_size':world,'summary':plan_summary(contexts,layout),'contexts':[c.__dict__ if hasattr(c,'__dict__') else {'sample_id':c.sample_id,'branch_ids':list(c.branch_ids),'gate_indices':list(c.gate_indices)} for c in contexts]},indent=2),encoding='utf-8')
    if a.dry_run:
        if world>1: torch.distributed.barrier(); torch.distributed.destroy_process_group()
        return
    device=f'cuda:{rank}'
    tok=tokenizer_loader(a.model_name_or_path,use_fast=True,trust_remote_code=True,local_files_only=True); mat=materializer_factory(tok,layout=layout)
    model=model_loader(a.model_name_or_path,torch_dtype=resolve_dtype(a.torch_dtype),attn_implementation=a.attn_implementation,trust_remote_code=True,local_files_only=True,low_cpu_mem_usage=True).to(device).eval(); model.config.use_cache=True
    raw=a.output_dir/f'raw_branch_scores.rank{rank:03d}.jsonl'; raw.parent.mkdir(parents=True, exist_ok=True); raw.touch(exist_ok=True); local=contexts[rank::world]
    print(f'[rank {rank}/{world}] contexts={len(local)} raw={raw}', flush=True)
    completed_local = 0
    local_total_branches = sum(len(context.branch_ids) for context in local)
    progress = LogProgressBar(
        total=max(1, len(local)),
        label='retrieval initialization',
        unit='rank-0 context slots',
        enabled=rank == 0,
    )
    progress.start(detail=f'{world} GPUs; {len(contexts)} global contexts')
    for context_index, context in enumerate(local, start=1):
        plan=tuple(read_manifest(Path(context.manifest_path)))[context.manifest_index]; sample=mat.materialize_independent_queries(plan); ids=torch.tensor(sample.prompt_input_ids,device=device).unsqueeze(0); cache=DynamicCache()
        try:
            with torch.inference_mode():
                out=model.model(input_ids=ids,past_key_values=cache,use_cache=True,output_attentions=False,return_dict=True); cache=out.past_key_values; del out
                if int(cache.get_seq_length())!=layout.context_tokens: raise AssertionError('bad prefill length')
                for suffix in sample.suffixes:
                    needle=plan.needles[suffix.needle_index]; started=time.perf_counter()
                    rec={'schema_version':RAW_SCHEMA_VERSION,'branch_id':branch_id(plan.sample_id,suffix.needle_index),'sample_id':plan.sample_id,'manifest_path':context.manifest_path,'manifest_index':context.manifest_index,'prompt_ids_sha256':sample.prompt_ids_sha256,'value_type':plan.value_type,'needle_count':plan.needle_count,'template_id':plan.template_id,'coverage_round':plan.coverage_round,'query_position':suffix.query_position,'needle_index':suffix.needle_index,'key':suffix.key,'value':suffix.value,'absolute_block':needle.absolute_block,'relative_distance':needle.relative_distance,'gate_index':needle.gate_index,'retrieval_k':a.retrieval_k,'error':None}
                    try:
                        rec.update(score_branch_fn(model=model,tokenizer=tok,cache=cache,prompt_length=layout.context_tokens,suffix=suffix,needle=needle,layout=layout,retrieval_k=a.retrieval_k)); append_jsonl(raw,rec)
                        completed_local += 1
                    finally: cache.crop(layout.context_tokens)
        finally: del ids,cache; torch.cuda.empty_cache(); gc.collect()
        progress.update(
            context_index,
            detail=f'rank0 branches={completed_local}/{local_total_branches}',
        )
    print(f'[rank {rank}] finished local branches={completed_local}', flush=True)
    if world>1:
        print(f'[rank {rank}] entering merge barrier', flush=True)
        torch.distributed.barrier()
        print(f'[rank {rank}] passed merge barrier', flush=True)
    if rank==0:
        records=[]
        for r in range(world): records.extend(read_jsonl(a.output_dir/f'raw_branch_scores.rank{r:03d}.jsonl'))
        records.sort(key=lambda x:x['branch_id']); append_path=a.output_dir/'raw_branch_scores.jsonl'; append_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records),encoding='utf-8')
        ids=[x['branch_id'] for x in records if x.get('error') is None]
        if len(ids)!=len(set(ids)) or len(ids)!=sum(len(c.branch_ids) for c in contexts): raise RuntimeError('missing, failed, or duplicate branches')
        agg=aggregate_raw_records_condition_balanced(records,num_distances=layout.num_learnable_blocks,top_q=a.top_q); good=[x for x in records if x.get('error') is None]; save_aggregate(agg,output_dir=a.output_dir,metadata={'world_size':world,'master_seed':a.master_seed,'num_successful_records':len(good),'coverage_round':a.coverage_round,'retrieval_k':a.retrieval_k,'aggregation':'equal_condition_round_after_within_condition_distance_mean'},min_distance=layout.min_learnable_distance)
    if world>1: torch.distributed.destroy_process_group()
