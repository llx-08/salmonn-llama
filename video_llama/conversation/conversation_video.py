"""
Conversation prompt template of Video-LLaMA.
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/conversation/conversation.py 
"""
import argparse
import time
from PIL import Image
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, WhisperFeatureExtractor
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
import os
from video_llama.common.registry import registry
from video_llama.processors.video_processor import ToTHWC,ToUint8,load_video
from video_llama.processors import Blip2ImageEvalProcessor
import soundfile as sf 
# from video_llama.models.ImageBind.data import load_and_transform_audio_data
class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

default_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
conv_llava_llama_2 = Conversation(
    system="You are a helpful language and audio assistant. "
           "You are able to understand the audio content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)
class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()
        # stop_words_ids = [torch.tensor([835]).to(self.device),
        #                   torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        # self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and ('</Video>' in conv.messages[-1][1] or '</Image>' in conv.messages[-1][1]):  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, user_message, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        speech_embeds = img_list[0]
        print("check speech embeds")
        print(speech_embeds.shape)        
        # embs = self.get_context_emb(conv, img_list)

        # current_max_len = embs.shape[1] + max_new_tokens
        # if current_max_len - max_length > 0:
        #     print('Warning: The number of tokens in current conversation exceeds the max length. '
        #           'The model will not see the contexts outside the range.')
        # begin_idx = max(0, current_max_len - max_length)

        # embs = embs[:, begin_idx:]

        """warp instruction with embedding"""
        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        embed_tokens = self.model.llama_model.model.embed_tokens

        prompt_pattern="USER: <Speech><ImageHere></Speech> {}\nASSISTANT:"
        prompt_left, prompts_right = prompt_pattern.format(user_message).split('<ImageHere>')

        print("check prompt")
        print(prompt_left)
        print(prompts_right)
        
        prompt_left_ids = self.model.llama_tokenizer(
            prompt_left,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids)
        prompt_right_ids = self.model.llama_tokenizer(
            prompts_right,
            return_tensors="pt",
            add_special_tokens=False
        ).to(speech_embeds.device).input_ids
        prompt_right_embeds = embed_tokens(prompt_right_ids)

        bos_embeds = self.model.llama_model.model.embed_tokens(
            torch.ones(
                [1, 1],
                dtype=torch.long,
                device=speech_embeds.device,
            ) * self.model.llama_tokenizer.bos_token_id
        )

        embeds = torch.cat([bos_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        
        if conv.sep =="###":
            stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # stopping_criteria
        output = self.model.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            attention_mask=atts,
            bos_token_id=self.model.llama_tokenizer.bos_token_id,
            eos_token_id=self.model.llama_tokenizer.eos_token_id,
            pad_token_id=self.model.llama_tokenizer.pad_token_id
        )
        # output_token = outputs[0]
        # if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        #     output_token = output_token[1:]
        # if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
        #     output_token = output_token[1:]
        # output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        # if conv.sep =="###":
        #     output_text = output_text.split('###')[0]  # remove the stop sign '###'
        #     output_text = output_text.split('Assistant:')[-1].strip()
        # else:
        #     output_text = output_text.split(conv.sep2)[0]  # remove the stop sign '###'
        #     output_text = output_text.split(conv.roles[1]+':')[-1].strip()
        # conv.messages[-1][1] = output_text
        # return output_text, output_token.cpu().numpy()
        
        output_text = self.model.llama_tokenizer.batch_decode(output, add_special_tokens=False, skip_special_tokens=True)

        return output_text
    
    def upload_video(self, video_path, conv, img_list):
        # msg = ""
        # if isinstance(video_path, str):  # is a video path
        #     ext = os.path.splitext(video_path)[-1].lower()
        #     print("video path")
        #     print(video_path)
        #     # image = self.vis_processor(image).unsqueeze(0).to(self.device)
        #     video, msg = load_video(
        #         video_path=video_path,
        #         n_frms=8,
        #         height=224,
        #         width=224,
        #         sampling ="uniform", return_msg = True
        #     )
        #     video = self.vis_processor.transform(video)
        #     video = video.unsqueeze(0).to(self.device)
        #     # print(image)
        # else:
        #     raise NotImplementedError
        
        # try:
        #     audio_flag = 1
        #     audio = load_and_transform_audio_data([video_path],"cpu",  clips_per_video=8)
        #     audio = audio.to(self.device)
        # except :
        #     print('no audio is found')
        #     audio_flag = 0
        # finally:
        #     if audio_flag == 1:
        #         # image_emb, _ = self.model.encode_videoQformer_audiovideo(video,audio)
        #         image_emb, _ = self.model.encode_videoQformer_visual(video)
        #         audio_emb,_  = self.model.encode_audioQformer(audio)
        #         img_list.append(audio_emb)
        #         img_list.append(image_emb)

        #         print("check audio&image embedding")
        #         print(img_list)

        #         conv.system = ""
        #         # conv.append_message(conv.roles[0], "The audio of this video is <Video><ImageHere></Video> ")
        #         conv.append_message(conv.roles[0], "Close your eyes, open your ears and you imagine only based on the sound that: <ImageHere>. \
        #         Close your ears, open your eyes and you see that <Video><ImageHere></Video>.  \
        #         Now answer my question based on what you have just seen and heard.")

        #     else:  # only vison no audio
        #         # conv.system = "You can understand the video that the user provides. Follow the instructions carefully and explain your answers in detail."
        #         print("only vison no audio")
        #         image_emb, _ = self.model.encode_videoQformer_visual(video)
        #         img_list.append(image_emb)
        #         conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
        #     return "Received."
        pass

    def upload_audio(self, audio_path, conv, img_list):
        #audio = load_and_transform_audio_data(audio_path, "cpu", clips_per_video=8)
        audio, sr = sf.read(audio_path)
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        if len(audio) > 30 * sr:
            audio = audio[: 30 * sr]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000, res_type="fft")
    
        audio_emb, atts_llama = self.model.encode_audioQformer(audio)

        img_list.append(audio_emb)

        conv.system = ""
        # conv.append_message(conv.roles[0], "The audio of this video is <Video><ImageHere></Video> ")
        conv.append_message(conv.roles[0], "{} Transcribe the audio".format('<ImageHere>'))

        return "Received."

    def upload_video_without_audio(self, video_path, conv, img_list):
        msg = ""
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            print(video_path)
            # image = self.vis_processor(image).unsqueeze(0).to(self.device)
            video, msg = load_video(
                video_path=video_path,
                n_frms=8,
                height=224,
                width=224,
                sampling ="uniform", return_msg = True
            )
            video = self.vis_processor.transform(video)
            video = video.unsqueeze(0).to(self.device)
            # print(image)
        else:
            raise NotImplementedError

        # conv.system = "You can understand the video that the user provides.  Follow the instructions carefully and explain your answers in detail."
        image_emb, _ = self.model.encode_videoQformer_visual(video)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
        return "Received."

    def upload_img(self, image, conv, img_list):

        msg = ""
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB') # 增加一个时间维度
            image = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        else:
            raise NotImplementedError

        image_emb, _ = self.model.encode_videoQformer_visual(image)
        img_list.append(image_emb)
        # Todo msg=""
        conv.append_message(conv.roles[0], "<Image><ImageHere></Image> "+ msg)

        return "Received."

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        
        print("get prompt:", prompt)
        
        prompt_segs = prompt.split('<ImageHere>')
        print(prompt_segs, len(img_list), img_list[0].size())
        #assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).input_ids.type(torch.long)
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t.to(self.device)) for seg_t in seg_tokens]
        print(seg_embs[0].size())
        
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

if __name__ =='__main__':
    video_path = '/mnt/workspace/videoGPT/Video-LLaMA/examples/applausing.mp4'
    # import torch.classes.torchaudio.ffmpeg_StreamReader
    # ffmpeg_StreamReader(video_path)
    load_and_transform_audio_data([video_path],"cpu",clips_per_video=8)
