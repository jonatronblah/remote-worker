import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union, Optional

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from huggingface_hub import hf_hub_download
from transformers import DynamicCache
from safetensors.torch import load_file as safe_load_file

from . import MIDI
from .midi_model import MIDIModel, config_name_list, MIDIModelConfig
from .midi_synthesizer import MidiSynthesizer
from .midi_tokenizer import MIDITokenizerV1, MIDITokenizerV2

MAX_SEED = np.iinfo(np.int32).max


class opt:
    device = "cuda"
    batch = 4


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt=None,
    batch_size=1,
    max_len=512,
    temp=1.0,
    top_p=0.98,
    top_k=20,
    disable_patch_change=False,
    disable_control_change=False,
    disable_channels=None,
    generator=None,
):
    if disable_channels is not None:
        disable_channels = [
            tokenizer.parameter_ids["channel"][c] for c in disable_channels
        ]
    else:
        disable_channels = []
    max_token_seq = tokenizer.max_token_seq
    if prompt is None:
        input_tensor = torch.full(
            (1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=model.device
        )
        input_tensor[0, 0] = tokenizer.bos_id  # bos
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = torch.cat([input_tensor] * batch_size, dim=0)
    else:
        if len(prompt.shape) == 2:
            prompt = prompt[None, :]
            prompt = np.repeat(prompt, repeats=batch_size, axis=0)
        elif prompt.shape[0] == 1:
            prompt = np.repeat(prompt, repeats=batch_size, axis=0)
        elif len(prompt.shape) != 3 or prompt.shape[0] != batch_size:
            raise ValueError(f"invalid shape for prompt, {prompt.shape}")
        prompt = prompt[..., :max_token_seq]
        if prompt.shape[-1] < max_token_seq:
            prompt = np.pad(
                prompt,
                ((0, 0), (0, 0), (0, max_token_seq - prompt.shape[-1])),
                mode="constant",
                constant_values=tokenizer.pad_id,
            )
        input_tensor = torch.from_numpy(prompt).to(
            dtype=torch.long, device=model.device
        )
    input_tensor = input_tensor[:, -4096:]
    cur_len = input_tensor.shape[1]
    bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
    cache1 = DynamicCache()
    past_len = 0
    with bar:
        while cur_len < max_len:
            end = [False] * batch_size
            hidden = model.forward(input_tensor[:, past_len:], cache=cache1)[:, -1]
            next_token_seq = None
            event_names = [""] * batch_size
            cache2 = DynamicCache()
            for i in range(max_token_seq):
                mask = torch.zeros(
                    (batch_size, tokenizer.vocab_size),
                    dtype=torch.int64,
                    device=model.device,
                )
                for b in range(batch_size):
                    if end[b]:
                        mask[b, tokenizer.pad_id] = 1
                        continue
                    if i == 0:
                        mask_ids = list(tokenizer.event_ids.values()) + [
                            tokenizer.eos_id
                        ]
                        if disable_patch_change:
                            mask_ids.remove(tokenizer.event_ids["patch_change"])
                        if disable_control_change:
                            mask_ids.remove(tokenizer.event_ids["control_change"])
                        mask[b, mask_ids] = 1
                    else:
                        param_names = tokenizer.events[event_names[b]]
                        if i > len(param_names):
                            mask[b, tokenizer.pad_id] = 1
                            continue
                        param_name = param_names[i - 1]
                        mask_ids = tokenizer.parameter_ids[param_name]
                        if param_name == "channel":
                            mask_ids = [
                                i for i in mask_ids if i not in disable_channels
                            ]
                        mask[b, mask_ids] = 1
                mask = mask.unsqueeze(1)
                x = next_token_seq
                if i != 0:
                    hidden = None
                    x = x[:, -1:]
                logits = model.forward_token(hidden, x, cache=cache2)[:, -1:]
                scores = torch.softmax(logits / temp, dim=-1) * mask
                samples = model.sample_top_p_k(
                    scores, top_p, top_k, generator=generator
                )
                if i == 0:
                    next_token_seq = samples
                    for b in range(batch_size):
                        if end[b]:
                            continue
                        eid = samples[b].item()
                        if eid == tokenizer.eos_id:
                            end[b] = True
                        else:
                            event_names[b] = tokenizer.id_events[eid]
                else:
                    next_token_seq = torch.cat([next_token_seq, samples], dim=1)
                    if all(
                        [
                            len(tokenizer.events[event_names[b]]) == i
                            for b in range(batch_size)
                            if not end[b]
                        ]
                    ):
                        break
            if next_token_seq.shape[1] < max_token_seq:
                next_token_seq = F.pad(
                    next_token_seq,
                    (0, max_token_seq - next_token_seq.shape[1]),
                    "constant",
                    value=tokenizer.pad_id,
                )
            next_token_seq = next_token_seq.unsqueeze(1)
            input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
            past_len = cur_len
            cur_len += 1
            bar.update(1)
            yield next_token_seq[:, 0].cpu().numpy()
            if all(end):
                break


def create_msg(name, data):
    return {"name": name, "data": data}


def send_msgs(msgs):
    return json.dumps(msgs)


number2drum_kits = {
    -1: "None",
    0: "Standard",
    8: "Room",
    16: "Power",
    24: "Electric",
    25: "TR-808",
    32: "Jazz",
    40: "Blush",
    48: "Orchestra",
}
patch2number = {v: k for k, v in MIDI.Number2patch.items()}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}
key_signatures = [
    "C♭",
    "A♭m",
    "G♭",
    "E♭m",
    "D♭",
    "B♭m",
    "A♭",
    "Fm",
    "E♭",
    "Cm",
    "B♭",
    "Gm",
    "F",
    "Dm",
    "C",
    "Am",
    "G",
    "Em",
    "D",
    "Bm",
    "A",
    "F♯m",
    "E",
    "C♯m",
    "B",
    "G♯m",
    "F♯",
    "D♯m",
    "C♯",
    "A♯m",
]


def run(
    model,
    tokenizer,
    OUTPUT_BATCH_SIZE,
    tab,
    mid_seq,
    continuation_state,
    continuation_select,
    instruments,
    drum_kit,
    bpm,
    time_sig,
    key_sig,
    mid,
    midi_events,
    reduce_cc_st,
    remap_track_channel,
    add_default_instr,
    remove_empty_channels,
    seed,
    seed_rand,
    gen_events,
    temp,
    top_p,
    top_k,
    allow_cc,
):
    bpm = int(bpm)
    if time_sig == "auto":
        time_sig = None
        time_sig_nn = 4
        time_sig_dd = 2
    else:
        time_sig_nn, time_sig_dd = time_sig.split("/")
        time_sig_nn = int(time_sig_nn)
        time_sig_dd = {2: 1, 4: 2, 8: 3}[int(time_sig_dd)]
    if key_sig == 0:
        key_sig = None
        key_sig_sf = 0
        key_sig_mi = 0
    else:
        key_sig = key_sig - 1
        key_sig_sf = key_sig // 2 - 7
        key_sig_mi = key_sig % 2
    gen_events = int(gen_events)
    max_len = gen_events
    if seed_rand:
        seed = np.random.randint(0, MAX_SEED)
    generator = torch.Generator(opt.device).manual_seed(seed)
    disable_patch_change = False
    disable_channels = None
    if tab == 0:
        i = 0
        mid = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
        if tokenizer.version == "v2":
            if time_sig is not None:
                mid.append(
                    tokenizer.event2tokens(
                        ["time_signature", 0, 0, 0, time_sig_nn - 1, time_sig_dd - 1]
                    )
                )
            if key_sig is not None:
                mid.append(
                    tokenizer.event2tokens(
                        ["key_signature", 0, 0, 0, key_sig_sf + 7, key_sig_mi]
                    )
                )
        if bpm != 0:
            mid.append(tokenizer.event2tokens(["set_tempo", 0, 0, 0, bpm]))
        patches = {}
        if instruments is None:
            instruments = []
        for instr in instruments:
            patches[i] = patch2number[instr]
            i = (i + 1) if i != 8 else 10
        if drum_kit != "None":
            patches[9] = drum_kits2number[drum_kit]
        for i, (c, p) in enumerate(patches.items()):
            mid.append(tokenizer.event2tokens(["patch_change", 0, 0, i + 1, c, p]))
        mid = np.asarray([mid] * OUTPUT_BATCH_SIZE, dtype=np.int64)
        mid_seq = mid.tolist()
        if len(instruments) > 0:
            disable_patch_change = True
            disable_channels = [i for i in range(16) if i not in patches]
    elif tab == 1 and mid is not None:
        eps = 4 if reduce_cc_st else 0
        mid = tokenizer.tokenize(
            MIDI.midi2score(mid),
            cc_eps=eps,
            tempo_eps=eps,
            remap_track_channel=remap_track_channel,
            add_default_instr=add_default_instr,
            remove_empty_channels=remove_empty_channels,
        )
        midi_events = int(midi_events)
        if midi_events <= 4096:
            mid = mid[:midi_events]
        mid = np.asarray([mid] * OUTPUT_BATCH_SIZE, dtype=np.int64)
        mid_seq = mid.tolist()
    elif tab == 2 and mid_seq is not None:
        mid = np.asarray(mid_seq, dtype=np.int64)
        if continuation_select > 0:
            continuation_state.append(mid_seq)
            mid = np.repeat(
                mid[continuation_select - 1 : continuation_select],
                repeats=OUTPUT_BATCH_SIZE,
                axis=0,
            )
            mid_seq = mid.tolist()
        else:
            continuation_state.append(mid.shape[1])
    else:
        continuation_state = [0]
        mid = [[tokenizer.bos_id] + [tokenizer.pad_id] * (tokenizer.max_token_seq - 1)]
        mid = np.asarray([mid] * OUTPUT_BATCH_SIZE, dtype=np.int64)
        mid_seq = mid.tolist()

    if mid is not None:
        max_len += mid.shape[1]

    init_msgs = [create_msg("progress", [0, gen_events])]
    if not (tab == 2 and continuation_select == 0):
        for i in range(OUTPUT_BATCH_SIZE):
            events = [tokenizer.tokens2event(tokens) for tokens in mid_seq[i]]
            init_msgs += [
                create_msg("visualizer_clear", [i, tokenizer.version]),
                create_msg("visualizer_append", [i, events]),
            ]
    yield mid_seq, continuation_state, seed
    midi_generator = generate(
        model=model,
        prompt=mid,
        batch_size=OUTPUT_BATCH_SIZE,
        max_len=max_len,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        disable_patch_change=disable_patch_change,
        disable_control_change=not allow_cc,
        disable_channels=disable_channels,
        generator=generator,
        tokenizer=tokenizer,
    )
    events = [list() for i in range(OUTPUT_BATCH_SIZE)]
    t = time.time()
    for i, token_seqs in enumerate(midi_generator):
        token_seqs = token_seqs.tolist()
        for j in range(OUTPUT_BATCH_SIZE):
            token_seq = token_seqs[j]
            mid_seq[j].append(token_seq)
            events[j].append(tokenizer.tokens2event(token_seq))
        if time.time() - t > 0.2:
            msgs = [create_msg("progress", [i + 1, gen_events])]
            for j in range(OUTPUT_BATCH_SIZE):
                msgs += [create_msg("visualizer_append", [j, events[j]])]
                events[j] = list()
            yield mid_seq, continuation_state, seed, send_msgs(msgs)
            t = time.time()
    yield mid_seq, continuation_state, seed


def finish_run(mid_seq, OUTPUT_BATCH_SIZE, tokenizer):
    if mid_seq is None:
        outputs = [None] * OUTPUT_BATCH_SIZE
        return *outputs, []
    outputs = []
    end_msgs = [create_msg("progress", [0, 0])]
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    for i in range(OUTPUT_BATCH_SIZE):
        events = [tokenizer.tokens2event(tokens) for tokens in mid_seq[i]]
        mid = tokenizer.detokenize(mid_seq[i])
        with open(f"outputs/output{i + 1}.mid", "wb") as f:
            f.write(MIDI.score2midi(mid))
        outputs.append(f"outputs/output{i + 1}.mid")
        end_msgs += [
            create_msg("visualizer_clear", [i, tokenizer.version]),
            create_msg("visualizer_append", [i, events]),
            create_msg("visualizer_end", i),
        ]
    return outputs


def load_model(path, model_config, lora_path):
    if model_config == "auto":
        config_path = Path(path).parent / "config.json"
        if config_path.exists():
            config = MIDIModelConfig.from_json_file(config_path)
        else:
            return "can not find config.json, please specify config"
    else:
        config = MIDIModelConfig.from_name(model_config)
    model = MIDIModel(config=config)
    tokenizer = model.tokenizer
    if path.endswith(".safetensors"):
        state_dict = safe_load_file(path)
    else:
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    if lora_path:
        model = model.load_merge_lora(lora_path)
    model.to(
        opt.device, dtype=torch.bfloat16 if opt.device == "cuda" else torch.float32
    ).eval()
    return model, tokenizer


def get_model_path():
    ckpt_files = glob.glob("**/*.ckpt", recursive=True)
    bin_files = glob.glob("**/*.bin", recursive=True)
    safetensors_files = glob.glob("**/*.safetensors", recursive=True)
    model_paths = sorted(ckpt_files + bin_files + safetensors_files)
    model_paths = [
        model_path for model_path in model_paths if "adapter_model" not in model_path
    ]  # lora
    return model_paths


def get_lora_path():
    lora_paths = sorted(glob.glob("**/adapter_config.json", recursive=True))
    lora_paths = [
        lora_path.replace("adapter_config.json", "") for lora_path in lora_paths
    ]
    return gr.Dropdown(choices=lora_paths)


def main(fileName):
    with open(fileName, mode="rb") as file:  # b is important -> binary
        midi_file = file.read()

    OUTPUT_BATCH_SIZE = 4
    soundfont_path = hf_hub_download(
        repo_id="skytnt/midi-model", filename="soundfont.sf2"
    )
    synthesizer = MidiSynthesizer(soundfont_path)
    thread_pool = ThreadPoolExecutor(max_workers=OUTPUT_BATCH_SIZE)
    model_paths = get_model_path()
    model, tokenizer = load_model(model_paths[1], "tv2o-medium", [])
    mid_seq = run(
        model,
        tokenizer,
        4,
        1,
        None,
        [0],
        0,
        None,
        None,
        0,
        "auto",
        0,
        midi_file,
        2048,
        True,
        True,
        True,
        True,
        0,
        True,
        2048,
        1.0,
        0.98,
        20,
        True,
    )
    outputs = finish_run(mid_seq, OUTPUT_BATCH_SIZE, tokenizer)
    return outputs
