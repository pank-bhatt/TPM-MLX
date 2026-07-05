# Copyright © 2026 TPM-MLX Authors. All rights reserved.

import os
import sys
import click
import logging
import uvicorn
from pathlib import Path

# Set up logging before loading other modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("tpm-mlx.cli")


@click.group()
def main():
    """TPM-MLX: Optimized Apple Silicon Inference Engine CLI"""
    pass


@main.command()
@click.option("--model", "-m", type=str, default="mlx-community/gemma-4-e2b-it-4bit", help="Model path or Hugging Face repository ID")
@click.option("--port", "-p", type=int, default=2505, help="Port to run the server on (default: 2505)")
@click.option("--host", "-h", type=str, default="127.0.0.1", help="Host address to run the server on (default: 127.0.0.1)")
@click.option("--max-kv-size", type=int, default=4096, help="Pre-allocated KV cache size (default: 4096)")
@click.option("--reasoning/--no-reasoning", default=False, help="Toggle whether the server outputs <think> blocks by default (default: False)")
def serve(model: str, port: int, host: str, max_kv_size: int, reasoning: bool):
    """Starts the FastAPI OpenAI-compatible server and Web Playground."""
    logger.info(f"Starting TPM-MLX REST server on http://{host}:{port}/")
    
    # Store settings in environment variables for server startup loading
    os.environ["TPM_DEFAULT_MODEL"] = model
    os.environ["TPM_MAX_KV_SIZE"] = str(max_kv_size)
    os.environ["TPM_DEFAULT_REASONING"] = str(reasoning)
    
    # Run Uvicorn ASGI server
    uvicorn.run("tpm_mlx.server:app", host=host, port=port, reload=False)


@main.command()
@click.option("--model", "-m", type=str, default="mlx-community/gemma-4-e2b-it-4bit", help="Model path or Hugging Face repository ID")
@click.option("--temp", "-t", type=float, default=0.0, help="Generation temperature (default: 0.0)")
@click.option("--max-tokens", "-n", type=int, default=4096, help="Maximum tokens to generate (default: 4096)")
@click.option("--max-kv-size", type=int, default=4096, help="Pre-allocated KV cache size (default: 4096)")
@click.option("--reasoning/--no-reasoning", default=False, help="Toggle displaying model reasoning <think> blocks (default: False)")
def chat(model: str, temp: float, max_tokens: int, max_kv_size: int, reasoning: bool):
    """Starts a local interactive chat session in the terminal."""
    click.echo(click.style(f"Initializing engine for {model}...", fg="cyan"))
    
    try:
        from tpm_mlx.engine import MLXEngine
        engine = MLXEngine(model_path_or_id=model, max_kv_size=max_kv_size)
    except Exception as e:
        click.echo(click.style(f"Error loading model: {e}", fg="red"), err=True)
        sys.exit(1)
        
    click.echo(click.style("Engine ready! Type '/exit' or '/quit' to exit.", fg="green"))
    click.echo(click.style(f"Settings: temp={temp}, max_tokens={max_tokens}, reasoning={reasoning}", fg="yellow"))
    click.echo("-" * 50)
    
    chat_history = []
    
    while True:
        try:
            user_input = click.prompt(click.style("You", fg="bright_blue"))
        except (KeyboardInterrupt, click.exceptions.Abort):
            click.echo("\nGoodbye!")
            break
            
        if user_input.strip() in ("/exit", "/quit"):
            click.echo("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        chat_history.append({"role": "user", "content": user_input})
        
        try:
            # Apply chat template
            prompt = engine.tokenizer.apply_chat_template(
                chat_history, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            from tpm_mlx.utils import apply_chat_template_fallback
            prompt = apply_chat_template_fallback(chat_history, engine.tokenizer)
            
        click.echo(click.style("Assistant: ", fg="bright_green"), nl=False)
        
        # Stream response
        buffer = ""
        full_assistant_text = ""
        inside_thought = False
        active_end_tag = None
        last_response = None
        
        start_tags = ["<think>", "<|channel>thought"]
        
        try:
            for response in engine.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temp,
                show_reasoning=reasoning
            ):
                last_response = response
                full_assistant_text += response.text
                
                if reasoning:
                    buffer += response.text
                    if not inside_thought:
                        # Look for any start tag
                        found_tag = None
                        found_idx = -1
                        for tag in start_tags:
                            idx = buffer.find(tag)
                            if idx != -1:
                                if found_idx == -1 or idx < found_idx:
                                    found_idx = idx
                                    found_tag = tag
                        
                        if found_tag is not None:
                            # Print text before tag
                            click.echo(buffer[:found_idx], nl=False)
                            # Print thought header in dimmed + italic
                            click.echo("\n\033[90m\033[3m[Thinking...]\n", nl=False)
                            buffer = buffer[found_idx + len(found_tag):]
                            inside_thought = True
                            active_end_tag = "</think>" if found_tag == "<think>" else "<channel|>"
                    else:
                        # Look for active end tag
                        idx = buffer.find(active_end_tag)
                        if idx != -1:
                            # Print thought content
                            click.echo(buffer[:idx], nl=False)
                            # Reset styles
                            click.echo("\033[0m\n", nl=False)
                            buffer = buffer[idx + len(active_end_tag):]
                            inside_thought = False
                            active_end_tag = None
                            
                    # Avoid splitting partial tags
                    keep_len = 0
                    target = active_end_tag if inside_thought else None
                    if target:
                        for i in range(1, len(target)):
                            if buffer.endswith(target[:i]):
                                keep_len = i
                                break
                    else:
                        for tag in start_tags:
                            for i in range(1, len(tag)):
                                if buffer.endswith(tag[:i]):
                                    keep_len = max(keep_len, i)
                                    break
                                    
                    if keep_len > 0:
                        print_text = buffer[:-keep_len]
                        buffer = buffer[-keep_len:]
                    else:
                        print_text = buffer
                        buffer = ""
                        
                    if print_text:
                        click.echo(print_text, nl=False)
                else:
                    # No reasoning (filtered at engine level)
                    click.echo(response.text, nl=False)
                    
            # Print remaining buffer content
            if buffer:
                click.echo(buffer, nl=False)
            if inside_thought:
                click.echo("\033[0m", nl=False) # Safely reset style if terminated inside thought
                
            click.echo() # Newline
            
            # Print performance metrics at the bottom
            if last_response:
                metrics_str = (
                    f"TPS: {last_response.generation_tps:.2f} tokens/s | "
                    f"TTFT: {last_response.prompt_tokens / last_response.prompt_tps * 1000.0:.2f} ms | "
                    f"Prompt: {last_response.prompt_tokens} tokens ({last_response.prompt_tps:.2f} tokens/s) | "
                    f"Generation: {last_response.generation_tokens} tokens"
                )
                click.echo(click.style(f"[{metrics_str}]", fg="cyan", dim=True))
                
                # Append full generated text (including <think> if reasoning is enabled) to history
                chat_history.append({"role": "assistant", "content": full_assistant_text})
                
        except Exception as e:
            click.echo(click.style(f"\nGeneration Error: {e}", fg="red"), err=True)
            
        click.echo("-" * 50)


@main.command()
@click.argument("model", type=str)
def download(model: str):
    """Downloads weight files from Hugging Face for a given repository ID."""
    click.echo(f"Resolving and downloading {model} from Hugging Face Hub...")
    try:
        from mlx_lm.utils import _download
        local_path = _download(model)
        click.echo(click.style(f"Download complete! Saved to {local_path}", fg="green"))
    except Exception as e:
        click.echo(click.style(f"Download failed: {e}", fg="red"), err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
