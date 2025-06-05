export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM
export CUDA_VISIBLE_DEVICES=2,3 # Use GPU 2 and 3
export NUMEXPR_MAX_THREADS=128 # Utilize all 128 cores for numerical computations

NUM_GPUS=2

#MODEL=data/OpenR1-Distill-0.6B
MODEL=Qwen/Qwen3-0.6B-Base
MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:8192,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

#MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# AIME 24
TASK=aime24    
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# Math 500  
TASK=math_500
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# LiveCodeBench
#lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
#    --use-chat-template \
#    --output-dir $OUTPUT_DIR 
