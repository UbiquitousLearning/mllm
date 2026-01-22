---
license: other
license_name: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/main/LICENSE
language:
- en
tags:
- multimodal
library_name: transformers
pipeline_tag: any-to-any
---

# Qwen2.5-Omni
<a href="https://chat.qwen.ai/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/%F0%9F%92%9C%EF%B8%8F%20Qwen%20Chat%20-536af5" style="display: inline-block; vertical-align: middle;"/>
</a>


## Overview 
### Introduction
Qwen2.5-Omni is an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. 

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/qwen_omni.png" width="80%"/>
<p>

### Key Features

* **Omni and Novel Architecture**: We propose Thinker-Talker architecture, an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. We propose a novel position embedding, named TMRoPE (Time-aligned Multimodal RoPE), to synchronize the timestamps of video inputs with audio.

* **Real-Time Voice and Video Chat**: Architecture designed for fully real-time interactions, supporting chunked input and immediate output.

* **Natural and Robust Speech Generation**: Surpassing many existing streaming and non-streaming alternatives, demonstrating superior robustness and naturalness in speech generation.

* **Strong Performance Across Modalities**: Exhibiting exceptional performance across all modalities when benchmarked against similarly sized single-modality models. Qwen2.5-Omni outperforms the similarly sized Qwen2-Audio in audio capabilities and achieves comparable performance to Qwen2.5-VL-7B.

* **Excellent End-to-End Speech Instruction Following**: Qwen2.5-Omni shows performance in end-to-end speech instruction following that rivals its effectiveness with text inputs, evidenced by benchmarks such as MMLU and GSM8K.

### Model Architecture

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/overview.png" width="80%"/>
<p>

### Performance

We conducted a comprehensive evaluation of Qwen2.5-Omni, which demonstrates strong performance across all modalities when compared to similarly sized single-modality models and closed-source models like Qwen2.5-VL-7B, Qwen2-Audio, and Gemini-1.5-pro. In tasks requiring the integration of multiple modalities, such as OmniBench, Qwen2.5-Omni achieves state-of-the-art performance. Furthermore, in single-modality tasks, it excels in areas including speech recognition (Common Voice), translation (CoVoST2), audio understanding (MMAU), image reasoning (MMMU, MMStar), video understanding (MVBench), and speech generation (Seed-tts-eval and subjective naturalness).

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/bar.png" width="80%"/>
<p>

<details>
<summary>Multimodality  -> Text</summary>

<table class="tg"><thead>
  <tr>
    <th class="tg-0lax">Datasets</th>
    <th class="tg-0lax">Model</th>
    <th class="tg-0lax">Performance</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0lax" rowspan="10">OmniBench<br>Speech | Sound Event | Music | Avg</td>
    <td class="tg-0lax">Gemini-1.5-Pro</td>
    <td class="tg-0lax">42.67%|42.26%|46.23%|42.91%</td>
  </tr>
  <tr>
    <td class="tg-0lax">MIO-Instruct</td>
    <td class="tg-0lax">36.96%|33.58%|11.32%|33.80%</td>
  </tr>
  <tr>
    <td class="tg-0lax">AnyGPT (7B)</td>
    <td class="tg-0lax">17.77%|20.75%|13.21%|18.04%</td>
  </tr>
  <tr>
    <td class="tg-0lax">video-SALMONN</td>
    <td class="tg-0lax">34.11%|31.70%|<strong>56.60%</strong>|35.64%</td>
  </tr>
  <tr>
    <td class="tg-0lax">UnifiedIO2-xlarge</td>
    <td class="tg-0lax">39.56%|36.98%|29.25%|38.00%</td>
  </tr>
  <tr>
    <td class="tg-0lax">UnifiedIO2-xxlarge</td>
    <td class="tg-0lax">34.24%|36.98%|24.53%|33.98%</td>
  </tr>
  <tr>
    <td class="tg-0lax">MiniCPM-o</td>
    <td class="tg-0lax">-|-|-|40.50%</td>
  </tr>
  <tr>
    <td class="tg-0lax">Baichuan-Omni-1.5</td>
    <td class="tg-0lax">-|-|-|42.90%</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">52.14%|52.08%|52.83%|52.19%</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax"><strong>55.25%</strong>|<strong>60.00%</strong>|52.83%|<strong>56.13%</strong></td>
  </tr>
</tbody></table>
</details>


<details>
<summary>Audio -> Text</summary>


<table class="tg"><thead>
  <tr>
    <th class="tg-0lax">Datasets</th>
    <th class="tg-0lax">Model</th>
    <th class="tg-0lax">Performance</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9j4x" colspan="3">ASR</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="12">Librispeech<br>dev-clean | dev other | test-clean | test-other</td>
    <td class="tg-0lax">SALMONN</td>
    <td class="tg-0lax">-|-|2.1|4.9</td>
  </tr>
  <tr>
    <td class="tg-0lax">SpeechVerse</td>
    <td class="tg-0lax">-|-|2.1|4.4</td>
  </tr>
  <tr>
    <td class="tg-0lax">Whisper-large-v3</td>
    <td class="tg-0lax">-|-|1.8|3.6</td>
  </tr>
  <tr>
    <td class="tg-0lax">Llama-3-8B</td>
    <td class="tg-0lax">-|-|-|3.4</td>
  </tr>
  <tr>
    <td class="tg-0lax">Llama-3-70B</td>
    <td class="tg-0lax">-|-|-|3.1</td>
  </tr>
  <tr>
    <td class="tg-0lax">Seed-ASR-Multilingual</td>
    <td class="tg-0lax">-|-|<strong>1.6</strong>|<strong>2.8</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax">MiniCPM-o</td>
    <td class="tg-0lax">-|-|1.7|-</td>
  </tr>
  <tr>
    <td class="tg-0lax">MinMo</td>
    <td class="tg-0lax">-|-|1.7|3.9</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen-Audio</td>
    <td class="tg-0lax">1.8|4.0|2.0|4.2</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2-Audio</td>
    <td class="tg-0lax"><strong>1.3</strong>|<strong>3.4</strong>|<strong>1.6</strong>|3.6</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">2.0|4.1|2.2|4.5</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax">1.6|3.5|1.8|3.4</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="5">Common Voice 15<br>en | zh | yue | fr</td>
    <td class="tg-0lax">Whisper-large-v3</td>
    <td class="tg-0lax">9.3|12.8|10.9|10.8</td>
  </tr>
  <tr>
    <td class="tg-0lax">MinMo</td>
    <td class="tg-0lax">7.9|6.3|6.4|8.5</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2-Audio</td>
    <td class="tg-0lax">8.6|6.9|<strong>5.9</strong>|9.6</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">9.1|6.0|11.6|9.6</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax"><strong>7.6</strong>|<strong>5.2</strong>|7.3|<strong>7.5</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="8">Fleurs<br>zh | en</td>
    <td class="tg-0lax">Whisper-large-v3</td>
    <td class="tg-0lax">7.7|4.1</td>
  </tr>
  <tr>
    <td class="tg-0lax">Seed-ASR-Multilingual</td>
    <td class="tg-0lax">-|<strong>3.4</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax">Megrez-3B-Omni</td>
    <td class="tg-0lax">10.8|-</td>
  </tr>
  <tr>
    <td class="tg-0lax">MiniCPM-o</td>
    <td class="tg-0lax">4.4|-</td>
  </tr>
  <tr>
    <td class="tg-0lax">MinMo</td>
    <td class="tg-0lax">3.0|3.8</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2-Audio</td>
    <td class="tg-0lax">7.5|-</td>
  </tr>
    <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">3.2|5.4</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax"><strong>3.0</strong>|4.1</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="6">Wenetspeech<br>test-net | test-meeting</td>
    <td class="tg-0lax">Seed-ASR-Chinese</td>
    <td class="tg-0lax"><strong>4.7|5.7</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax">Megrez-3B-Omni</td>
    <td class="tg-0lax">-|16.4</td>
  </tr>
  <tr>
    <td class="tg-0lax">MiniCPM-o</td>
    <td class="tg-0lax">6.9|-</td>
  </tr>
  <tr>
    <td class="tg-0lax">MinMo</td>
    <td class="tg-0lax">6.8|7.4</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">6.3|8.1</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax">5.9|7.7</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="4">Voxpopuli-V1.0-en</td>
    <td class="tg-0lax">Llama-3-8B</td>
    <td class="tg-0lax">6.2</td>
  </tr>
  <tr>
    <td class="tg-0lax">Llama-3-70B</td>
    <td class="tg-0lax"><strong>5.7</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">6.6</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax">5.8</td>
  </tr>
  <tr>
    <td class="tg-9j4x" colspan="3">S2TT</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="9">CoVoST2<br>en-de | de-en | en-zh | zh-en</td>
    <td class="tg-0lax">SALMONN</td>
    <td class="tg-0lax">18.6|-|33.1|-</td>
  </tr>
  <tr>
    <td class="tg-0lax">SpeechLLaMA</td>
    <td class="tg-0lax">-|27.1|-|12.3</td>
  </tr>
  <tr>
    <td class="tg-0lax">BLSP</td>
    <td class="tg-0lax">14.1|-|-|-</td>
  </tr>
  <tr>
    <td class="tg-0lax">MiniCPM-o</td>
    <td class="tg-0lax">-|-|<strong>48.2</strong>|27.2</td>
  </tr>
  <tr>
    <td class="tg-0lax">MinMo</td>
    <td class="tg-0lax">-|<strong>39.9</strong>|46.7|26.0</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen-Audio</td>
    <td class="tg-0lax">25.1|33.9|41.5|15.7</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2-Audio</td>
    <td class="tg-0lax">29.9|35.2|45.2|24.4</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">28.3|38.1|41.4|26.6</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax"><strong>30.2</strong>|37.7|41.4|<strong>29.4</strong></td>
  </tr>
  <tr>
    <td class="tg-9j4x" colspan="3">SER</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="6">Meld</td>
    <td class="tg-0lax">WavLM-large</td>
    <td class="tg-0lax">0.542</td>
  </tr>
  <tr>
    <td class="tg-0lax">MiniCPM-o</td>
    <td class="tg-0lax">0.524</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen-Audio</td>
    <td class="tg-0lax">0.557</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2-Audio</td>
    <td class="tg-0lax">0.553</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">0.558</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax"><strong>0.570</strong></td>
  </tr>
  <tr>
    <td class="tg-9j4x" colspan="3">VSC</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="6">VocalSound</td>
    <td class="tg-0lax">CLAP</td>
    <td class="tg-0lax">0.495</td>
  </tr>
  <tr>
    <td class="tg-0lax">Pengi</td>
    <td class="tg-0lax">0.604</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen-Audio</td>
    <td class="tg-0lax">0.929</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2-Audio</td>
    <td class="tg-0lax"><strong>0.939</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">0.936</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax"><strong>0.939</strong></td>
  </tr>
  <tr>
    <td class="tg-9j4x" colspan="3">Music</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="3">GiantSteps Tempo</td>
    <td class="tg-0lax">Llark-7B</td>
    <td class="tg-0lax">0.86</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax"><strong>0.88</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax"><strong>0.88</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="3">MusicCaps</td>
    <td class="tg-0lax">LP-MusicCaps</td>
    <td class="tg-0lax">0.291|0.149|0.089|<strong>0.061</strong>|0.129|0.130</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">0.325|<strong>0.163</strong>|<strong>0.093</strong>|0.057|<strong>0.132</strong>|<strong>0.229</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax"><strong>0.328</strong>|0.162|0.090|0.055|0.127|0.225</td>
  </tr>
  <tr>
    <td class="tg-9j4x" colspan="3">Audio Reasoning</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="4">MMAU<br>Sound | Music | Speech | Avg</td>
    <td class="tg-0lax">Gemini-Pro-V1.5</td>
    <td class="tg-0lax">56.75|49.40|58.55|54.90</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2-Audio</td>
    <td class="tg-0lax">54.95|50.98|42.04|49.20</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax"><strong>70.27</strong>|60.48|59.16|63.30</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax">67.87|<strong>69.16|59.76|65.60</strong></td>
  </tr>
  <tr>
    <td class="tg-9j4x" colspan="3">Voice Chatting</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="9">VoiceBench<br>AlpacaEval | CommonEval | SD-QA | MMSU</td>
    <td class="tg-0lax">Ultravox-v0.4.1-LLaMA-3.1-8B</td>
    <td class="tg-0lax"><strong>4.55</strong>|3.90|53.35|47.17</td>
  </tr>
  <tr>
    <td class="tg-0lax">MERaLiON</td>
    <td class="tg-0lax">4.50|3.77|55.06|34.95</td>
  </tr>
  <tr>
    <td class="tg-0lax">Megrez-3B-Omni</td>
    <td class="tg-0lax">3.50|2.95|25.95|27.03</td>
  </tr>
  <tr>
    <td class="tg-0lax">Lyra-Base</td>
    <td class="tg-0lax">3.85|3.50|38.25|49.74</td>
  </tr>
  <tr>
    <td class="tg-0lax">MiniCPM-o</td>
    <td class="tg-0lax">4.42|<strong>4.15</strong>|50.72|54.78</td>
  </tr>
  <tr>
    <td class="tg-0lax">Baichuan-Omni-1.5</td>
    <td class="tg-0lax">4.50|4.05|43.40|57.25</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2-Audio</td>
    <td class="tg-0lax">3.74|3.43|35.71|35.72</td>
  </tr>
    <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">4.32|4.00|49.37|50.23</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax">4.49|3.93|<strong>55.71</strong>|<strong>61.32</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="9">VoiceBench<br>OpenBookQA | IFEval | AdvBench | Avg</td>
    <td class="tg-0lax">Ultravox-v0.4.1-LLaMA-3.1-8B</td>
    <td class="tg-0lax">65.27|<strong>66.88</strong>|98.46|71.45</td>
  </tr>
  <tr>
    <td class="tg-0lax">MERaLiON</td>
    <td class="tg-0lax">27.23|62.93|94.81|62.91</td>
  </tr>
  <tr>
    <td class="tg-0lax">Megrez-3B-Omni</td>
    <td class="tg-0lax">28.35|25.71|87.69|46.25</td>
  </tr>
  <tr>
    <td class="tg-0lax">Lyra-Base</td>
    <td class="tg-0lax">72.75|36.28|59.62|57.66</td>
  </tr>
  <tr>
    <td class="tg-0lax">MiniCPM-o</td>
    <td class="tg-0lax">78.02|49.25|97.69|71.69</td>
  </tr>
  <tr>
    <td class="tg-0lax">Baichuan-Omni-1.5</td>
    <td class="tg-0lax">74.51|54.54|97.31|71.14</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2-Audio</td>
    <td class="tg-0lax">49.45|26.33|96.73|55.35</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B</td>
    <td class="tg-0lax">74.73|42.10|98.85|68.81</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B</td>
    <td class="tg-0lax"><strong>81.10</strong>|52.87|<strong>99.42</strong>|<strong>74.12</strong></td>
  </tr>
</tbody></table>
</details>

<details>
<summary>Image -> Text</summary>

| Dataset                        | Qwen2.5-Omni-7B | Qwen2.5-Omni-3B | Other Best | Qwen2.5-VL-7B | GPT-4o-mini | 
|--------------------------------|--------------|------------|------------|---------------|-------------|
| MMMU<sub>val</sub>             | 59.2         | 53.1       | 53.9       | 58.6          | **60.0**    | 
| MMMU-Pro<sub>overall</sub>     | 36.6         | 29.7       | -          | **38.3**      | 37.6        | 
| MathVista<sub>testmini</sub>   | 67.9         | 59.4       | **71.9**   | 68.2          | 52.5        | 
| MathVision<sub>full</sub>      | 25.0         | 20.8       | 23.1       | **25.1**      | -           | 
| MMBench-V1.1-EN<sub>test</sub> | 81.8         | 77.8       | 80.5       | **82.6**      | 76.0        | 
| MMVet<sub>turbo</sub>          | 66.8         | 62.1       | **67.5**   | 67.1          | 66.9        | 
| MMStar                         | **64.0**     | 55.7       | **64.0**   | 63.9          | 54.8        | 
| MME<sub>sum</sub>              | 2340         | 2117       | **2372**   | 2347          | 2003        | 
| MuirBench                      | 59.2         | 48.0       | -          | **59.2**      | -           | 
| CRPE<sub>relation</sub>        | **76.5**     | 73.7       | -          | 76.4          | -           | 
| RealWorldQA<sub>avg</sub>      | 70.3         | 62.6       | **71.9**   | 68.5          | -           | 
| MME-RealWorld<sub>en</sub>     | **61.6**     | 55.6       | -          | 57.4          | -           | 
| MM-MT-Bench                    | 6.0          | 5.0        | -          | **6.3**       | -           | 
| AI2D                           | 83.2         | 79.5       | **85.8**   | 83.9          | -           | 
| TextVQA<sub>val</sub>          | 84.4         | 79.8       | 83.2       | **84.9**      | -           | 
| DocVQA<sub>test</sub>          | 95.2         | 93.3       | 93.5       | **95.7**      | -           | 
| ChartQA<sub>test Avg</sub>     | 85.3         | 82.8       | 84.9       | **87.3**      | -           | 
| OCRBench_V2<sub>en</sub>       | **57.8**     | 51.7       | -          | 56.3          | -           | 


| Dataset                  | Qwen2.5-Omni-7B | Qwen2.5-Omni-3B | Qwen2.5-VL-7B | Grounding DINO | Gemini 1.5 Pro | 
|--------------------------|--------------|---------------|---------------|----------------|----------------|
| Refcoco<sub>val</sub>    | 90.5         | 88.7          | 90.0          | **90.6**       | 73.2           | 
| Refcoco<sub>textA</sub>  | **93.5**     | 91.8          | 92.5          | 93.2           | 72.9           | 
| Refcoco<sub>textB</sub>  | 86.6         | 84.0          | 85.4          | **88.2**       | 74.6           | 
| Refcoco+<sub>val</sub>   | 85.4         | 81.1          | 84.2          | **88.2**       | 62.5           | 
| Refcoco+<sub>textA</sub> | **91.0**     | 87.5          | 89.1          | 89.0           | 63.9           | 
| Refcoco+<sub>textB</sub> | **79.3**     | 73.2          | 76.9          | 75.9           | 65.0           | 
| Refcocog+<sub>val</sub>  | **87.4**     | 85.0          | 87.2          | 86.1           | 75.2           | 
| Refcocog+<sub>test</sub> | **87.9**     | 85.1          | 87.2          | 87.0           | 76.2           | 
| ODinW                    | 42.4         | 39.2          | 37.3          | **55.0**       | 36.7           | 
| PointGrounding           | 66.5         | 46.2          | **67.3**      | -              | -              | 
</details>


<details>
<summary>Video(without audio) -> Text</summary>

| Dataset                     | Qwen2.5-Omni-7B | Qwen2.5-Omni-3B | Other Best | Qwen2.5-VL-7B | GPT-4o-mini | 
|-----------------------------|--------------|------------|------------|---------------|-------------|
| Video-MME<sub>w/o sub</sub> | 64.3         | 62.0       | 63.9       | **65.1**      | 64.8        | 
| Video-MME<sub>w sub</sub>   | **72.4**     | 68.6       | 67.9       | 71.6          | -           | 
| MVBench                     | **70.3**     | 68.7       | 67.2       | 69.6          | -           | 
| EgoSchema<sub>test</sub>    | **68.6**     | 61.4       | 63.2       | 65.0          | -           | 
</details>

<details>
<summary>Zero-shot Speech Generation</summary>


<table class="tg"><thead>
  <tr>
    <th class="tg-0lax">Datasets</th>
    <th class="tg-0lax">Model</th>
    <th class="tg-0lax">Performance</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9j4x" colspan="3">Content Consistency</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="11">SEED<br>test-zh | test-en | test-hard </td>
    <td class="tg-0lax">Seed-TTS_ICL</td>
    <td class="tg-0lax">1.11 | 2.24 | 7.58</td>
  </tr>
  <tr>
    <td class="tg-0lax">Seed-TTS_RL</td>
    <td class="tg-0lax"><strong>1.00</strong> | 1.94 | <strong>6.42</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax">MaskGCT</td>
    <td class="tg-0lax">2.27 | 2.62 | 10.27</td>
  </tr>
  <tr>
    <td class="tg-0lax">E2_TTS</td>
    <td class="tg-0lax">1.97 | 2.19 | -</td>
  </tr>
  <tr>
    <td class="tg-0lax">F5-TTS</td>
    <td class="tg-0lax">1.56 | <strong>1.83</strong> | 8.67</td>
  </tr>
  <tr>
    <td class="tg-0lax">CosyVoice 2</td>
    <td class="tg-0lax">1.45 | 2.57 | 6.83</td>
  </tr>
  <tr>
    <td class="tg-0lax">CosyVoice 2-S</td>
    <td class="tg-0lax">1.45 | 2.38 | 8.08</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B_ICL</td>
    <td class="tg-0lax">1.95 | 2.87 | 9.92</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B_RL</td>
    <td class="tg-0lax">1.58 | 2.51 | 7.86</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B_ICL</td>
    <td class="tg-0lax">1.70 | 2.72 | 7.97</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B_RL</td>
    <td class="tg-0lax">1.42 | 2.32 | 6.54</td>
  </tr>
  <tr>
    <td class="tg-9j4x" colspan="3">Speaker Similarity</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="11">SEED<br>test-zh | test-en | test-hard </td>
    <td class="tg-0lax">Seed-TTS_ICL</td>
    <td class="tg-0lax">0.796 | 0.762 | 0.776</td>
  </tr>
  <tr>
    <td class="tg-0lax">Seed-TTS_RL</td>
    <td class="tg-0lax"><strong>0.801</strong> | <strong>0.766</strong> | <strong>0.782</strong></td>
  </tr>
  <tr>
    <td class="tg-0lax">MaskGCT</td>
    <td class="tg-0lax">0.774 | 0.714 | 0.748</td>
  </tr>
  <tr>
    <td class="tg-0lax">E2_TTS</td>
    <td class="tg-0lax">0.730 | 0.710 | -</td>
  </tr>
  <tr>
    <td class="tg-0lax">F5-TTS</td>
    <td class="tg-0lax">0.741 | 0.647 | 0.713</td>
  </tr>
  <tr>
    <td class="tg-0lax">CosyVoice 2</td>
    <td class="tg-0lax">0.748 | 0.652 | 0.724</td>
  </tr>
  <tr>
    <td class="tg-0lax">CosyVoice 2-S</td>
    <td class="tg-0lax">0.753 | 0.654 | 0.732</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B_ICL</td>
    <td class="tg-0lax">0.741 | 0.635 | 0.748</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-3B_RL</td>
    <td class="tg-0lax">0.744 | 0.635 | 0.746</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B_ICL</td>
    <td class="tg-0lax">0.752 | 0.632 | 0.747</td>
  </tr>
  <tr>
    <td class="tg-0lax">Qwen2.5-Omni-7B_RL</td>
    <td class="tg-0lax">0.754 | 0.641 | 0.752</td>
  </tr>
</tbody></table>
</details>

<details>
<summary>Text -> Text</summary>

| Dataset                           | Qwen2.5-Omni-7B | Qwen2.5-Omni-3B | Qwen2.5-7B | Qwen2.5-3B | Qwen2-7B | Llama3.1-8B | Gemma2-9B | 
|-----------------------------------|-----------|------------|------------|------------|------------|-------------|-----------|
| MMLU-Pro                          | 47.0      | 40.4       | **56.3**   | 43.7       | 44.1       | 48.3        | 52.1      | 
| MMLU-redux                        | 71.0      | 60.9       | **75.4**   | 64.4       | 67.3       | 67.2        | 72.8      | 
| LiveBench<sub>0831</sub>          | 29.6      | 22.3       | **35.9**   | 26.8       | 29.2       | 26.7        | 30.6      | 
| GPQA                              | 30.8      | 34.3       | **36.4**   | 30.3       | 34.3       | 32.8        | 32.8      | 
| MATH                              | 71.5      | 63.6       | **75.5**   | 65.9       | 52.9       | 51.9        | 44.3      | 
| GSM8K                             | 88.7      | 82.6       | **91.6**   | 86.7       | 85.7       | 84.5        | 76.7      | 
| HumanEval                         | 78.7      | 70.7       | **84.8**   |	74.4       | 79.9       | 72.6        | 68.9      | 
| MBPP                              | 73.2      | 70.4       | **79.2**   | 72.7       | 67.2       | 69.6        | 74.9      | 
| MultiPL-E                         | 65.8      | 57.6       | **70.4**   | 60.2       | 59.1       | 50.7        | 53.4      | 
| LiveCodeBench<sub>2305-2409</sub> | 24.6      | 16.5       | **28.7**   | 19.9       | 23.9       | 8.3         | 18.9      | 
</details>

## Quickstart

Below, we provide simple examples to show how to use Qwen2.5-Omni with 🤗 Transformers. The codes of Qwen2.5-Omni has been in the latest Hugging face transformers and we advise you to build from source with command:
```
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip install accelerate
```
or you might encounter the following error:
```
KeyError: 'qwen2_5_omni'
```


We offer a toolkit to help you handle various types of audio and visual input more conveniently, as if you were using an API. This includes base64, URLs, and interleaved audio, images and videos. You can install it using the following command and make sure your system has `ffmpeg` installed:

```bash
# It's highly recommended to use `[decord]` feature for faster video loading.
pip install qwen-omni-utils[decord] -U
```

If you are not using Linux, you might not be able to install `decord` from PyPI. In that case, you can use `pip install qwen-omni-utils -U` which will fall back to using torchvision for video processing. However, you can still [install decord from source](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source) to get decord used when loading video.

### 🤗  Transformers Usage

Here we show a code snippet to show you how to use the chat model with `transformers` and `qwen_omni_utils`:

```python
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-Omni-7B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
        ],
    },
]

# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
```

<details>
<summary>Minimum GPU memory requirements</summary>

|Model | Precision | 15(s) Video | 30(s) Video | 60(s) Video |
|--------------|-----------| ------------- | ------------- | ------------------ |
| Qwen-Omni-3B | FP32      | 89.10 GB      | Not Recommend | Not Recommend      |
| Qwen-Omni-3B | BF16      | 18.38 GB      | 22.43 GB      | 28.22 GB           |
| Qwen-Omni-7B | FP32      | 93.56 GB      | Not Recommend | Not Recommend      |
| Qwen-Omni-7B | BF16      | 31.11 GB      | 41.85 GB      | 60.19 GB           |

Note: The table above presents the theoretical minimum memory requirements for inference with `transformers` and `BF16` is test with `attn_implementation="flash_attention_2"`; however, in practice, the actual memory usage is typically at least 1.2 times higher. For more information, see the linked resource [here](https://huggingface.co/docs/accelerate/main/en/usage_guides/model_size_estimator).
</details>  

<details>
<summary>Video URL resource usage</summary>

Video URL compatibility largely depends on the third-party library version. The details are in the table below. Change the backend by `FORCE_QWENVL_VIDEO_READER=torchvision` or `FORCE_QWENVL_VIDEO_READER=decord` if you prefer not to use the default one.

| Backend     | HTTP | HTTPS |
|-------------|------|-------|
| torchvision >= 0.19.0 | ✅  | ✅   |
| torchvision < 0.19.0  | ❌  | ❌   |
| decord      | ✅  | ❌   |
</details>

<details>
<summary>Batch inference</summary>

The model can batch inputs composed of mixed samples of various types such as text, images, audio and videos as input when `return_audio=False` is set. Here is an example.

```python
# Sample messages for batch inference

# Conversation with video only
conversation1 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
        ]
    }
]

# Conversation with audio only
conversation2 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "/path/to/audio.wav"},
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": "who are you?"
    }
]


# Conversation with mixed media
conversation4 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/image.jpg"},
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "audio", "audio": "/path/to/audio.wav"},
            {"type": "text", "text": "What are the elements can you see and hear in these medias?"},
        ],
    }
]

# Combine messages for batch processing
conversations = [conversation1, conversation2, conversation3, conversation4]

# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for batch inference
text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)

inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Batch Inference
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
```
</details>

### Usage Tips

#### Prompt for audio output
If users need audio output, the system prompt must be set as "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.", otherwise the audio output may not work as expected.
```
{
    "role": "system",
    "content": [
        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
    ],
}
```
#### Use audio in video
In the process of multimodal interaction, the videos provided by users are often accompanied by audio (such as questions about the content in the video, or sounds generated by certain events in the video). This information is conducive to the model providing a better interactive experience. So we provide the following options for users to decide whether to use audio in video.
```python
# first place, in data preprocessing
audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)
```
```python
# second place, in model processor
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", 
                   padding=True, use_audio_in_video=True)
```
```python
#  third place, in model inference
text_ids, audio = model.generate(**inputs, use_audio_in_video=True)
```
It is worth noting that during a multi-round conversation, the `use_audio_in_video` parameter in these places must be set to the same, otherwise unexpected results will occur.

#### Use audio output or not

The model supports both text and audio outputs, if users do not need audio outputs, they can call `model.disable_talker()` after init the model. This option will save about `~2GB` of GPU memory but the `return_audio` option for `generate` function will only allow to be set at `False`.
```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto"
)
model.disable_talker()
```

In order to obtain a flexible experience, we recommend that users can decide whether to return audio when `generate` function is called. If `return_audio` is set to `False`, the model will only return text outputs to get text responses faster.

```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto"
)
...
text_ids = model.generate(**inputs, return_audio=False)
```

#### Change voice type of output audio
Qwen2.5-Omni supports the ability to change the voice of the output audio. The `"Qwen/Qwen2.5-Omni-7B"` checkpoint support two voice types as follow:

| Voice Type | Gender | Description |
|------------|--------|-------------|
| Chelsie    | Female | A honeyed, velvety voice that carries a gentle warmth and luminous clarity.|
| Ethan      | Male   | A bright, upbeat voice with infectious energy and a warm, approachable vibe.|

Users can use the `speaker` parameter of `generate` function to specify the voice type. By default, if `speaker` is not specified, the default voice type is `Chelsie`.

```python
text_ids, audio = model.generate(**inputs, speaker="Chelsie")
```

```python
text_ids, audio = model.generate(**inputs, speaker="Ethan")
```

#### Flash-Attention 2 to speed up generation

First, make sure to install the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

To load and run a model using FlashAttention-2, add `attn_implementation="flash_attention_2"` when loading the model:

```python
from transformers import Qwen2_5OmniForConditionalGeneration

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```


## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)



```BibTeX

@article{Qwen2.5-Omni,
  title={Qwen2.5-Omni Technical Report},
  author={Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, Junyang Lin},
  journal={arXiv preprint arXiv:2503.20215},
  year={2025}
}
```

<br>
