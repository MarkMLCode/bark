from typing import Dict, Optional, Union

import numpy as np

from .generation import codec_decode, generate_coarse, generate_fine, generate_text_semantic, InterruptEvent
import time
import threading

def text_to_semantic(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
    start_time: float = 0, 
    interrupt_event: InterruptEvent = None,
    max_gen_duration_s: float = None,
    top_k: int = None,
    top_p: float = None
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True,
        max_gen_duration_s = max_gen_duration_s,
        top_k = top_k,
        top_p = top_p,
        start_time = start_time, 
        interrupt_event = interrupt_event
    )
    return x_semantic


def semantic_to_waveform(
    semantic_tokens: np.ndarray,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens: semantic token output from `text_to_semantic`
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    # Return on bark interrupt
    #if interrupt_event is not None and convo_creation_time < interrupt_event.get_interrupt_time():
    #    return None

    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
        temp=0.5
    )
    # Return on bark interrupt
    #if interrupt_event is not None and convo_creation_time < interrupt_event.get_interrupt_time():
    #    return None
    
    audio_arr = codec_decode(fine_tokens)
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr


def save_as_prompt(filepath, full_generation):
    assert(filepath.endswith(".npz"))
    assert(isinstance(full_generation, dict))
    assert("semantic_prompt" in full_generation)
    assert("coarse_prompt" in full_generation)
    assert("fine_prompt" in full_generation)
    np.savez(filepath, **full_generation)

def split_array(arr, allowed_token):
    closest_index = len(arr) // 2  # start from middle
    while closest_index < len(arr):
        if arr[closest_index] in allowed_token:
            while closest_index > 0 and arr[closest_index - 1] in allowed_token:
                closest_index -= 1  # backtrack for repeating allowed tokens
            return arr[:closest_index], arr[closest_index:]
        closest_index += 1

    closest_index = len(arr) // 2  # start from middle again
    while closest_index >= 0:
        if arr[closest_index] in allowed_token:
            while closest_index > 0 and arr[closest_index - 1] in allowed_token:
                closest_index -= 1  # backtrack for repeating allowed tokens
            return arr[:closest_index], arr[closest_index:]
        closest_index -= 1

    # If there is no allowed_token in the array
    return arr, []

# Interrupt after time_before_interrupt sec, should interrupt whatever if not already finished
def interrupt_if_limit(interrupt_event: InterruptEvent, time_before_interrupt:float):
    time.sleep(time_before_interrupt)
    interrupt_event.interrupt()

def get_semantic_with_interrupt(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    text_temp: float = 0.7,
    silent: bool = False,
    sound_mode: bool = False,
    sound_mode_limit: float = None,
    sound_mode_retry_count: int = 0,
    max_gen_duration_s: float = None,
    top_k: int = None,
    top_p: float = None
):
    # When sound mode = True, will interrupt semantics after x sec if not already done
    if sound_mode and sound_mode_limit is not None : 
        interrupt_event = None
        start_semantic_time = 0

        # Retry sound_mode_retry_count - 1 times max, then take the last one no matter what
        for x in range(0,sound_mode_retry_count + 1):
            # No interrupt after x failed attemps
            if x == sound_mode_retry_count:
                interrupt_event = None
                start_semantic_time = 0
            else:
                interrupt_event = InterruptEvent()
                start_semantic_time = time.time()

                interrupt_thread = threading.Thread(target=interrupt_if_limit, args=(interrupt_event, sound_mode_limit))
                interrupt_thread.start()
            
            semantic_tokens = text_to_semantic(
                text,
                history_prompt=history_prompt,
                temp=text_temp,
                silent=silent,
                max_gen_duration_s = max_gen_duration_s,
                top_k = top_k,
                top_p = top_p,
                start_time = start_semantic_time,
                interrupt_event = interrupt_event,
            )

            if semantic_tokens is not None:
                print(f"Semantics found in : {time.time() - start_semantic_time} s")
                break
            else:
                print(f"Semantics interrupted after : {time.time() - start_semantic_time} s")
    else:
        semantic_tokens = text_to_semantic(
            text,
            history_prompt=history_prompt,
            temp=text_temp,
            silent=silent,
            max_gen_duration_s=max_gen_duration_s,
            top_k = top_k,
            top_p = top_p
        )

    return semantic_tokens

def generate_audio(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
    sound_mode: bool = False,
    sound_mode_limit: float = None,
    sound_mode_retry_count: int = 0,
    max_gen_duration_s: float = None,
    top_k: int = None,
    top_p: float = None,
    
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    
    start_section_time = time.time()

    semantic_tokens = get_semantic_with_interrupt(text, history_prompt, text_temp, silent, sound_mode, sound_mode_limit, sound_mode_retry_count, max_gen_duration_s, top_k, top_p)

    print(f'Semantic section : Time : {(time.time() - start_section_time)} s')

    # Exit immediately on interrupt
    #if interrupt_event is not None and convo_creation_time < interrupt_event.get_interrupt_time():
    #    return (None, None) if output_full else None

    section_time = time.time()
    out = semantic_to_waveform(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full
    )

    # Exit immediately on interrupt
    #if interrupt_event is not None and convo_creation_time < interrupt_event.get_interrupt_time():
    #    return (None, None) if output_full else None

    print(f'Waveform : Time: {(time.time() - section_time)} s')
    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    else:
        audio_arr = out
    return audio_arr