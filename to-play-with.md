# Running and Tuning LLMs on a Local Device

To date, large language models (LLMs) have typically been offered as a remote service, or required powerful machines on which to run them.

In distance education and open education settings, this can provide a barrier to access:

- if services are offered remotely, learners are tethered by the requirement of maintaining a network connection and cannot necessarily work offline;
- for local running, learners require access to a powerful computer, and potential a computer with a GPU.

Recent (March, 2023) demonstrations have show that it is possible to run LLMs on relatively small machines, albeit with limited performance.

One issue with using off the shelf LLMs is that they are "general purpose" in the sense that they have been trained on a wide range of generic texts.

In a learning context, we typically present materials that relate to a specific domain. Whilst we may still be able to admit of a certain amount of unreliablity in presented texts (knowing that a narrator is potentially unreliable may require us to be take a more critical stance towards the texts we are presented with), we might also prefer to provide learners with access to models that are more likely to recall or summarise, rather than hallucinate, answers to particular questions of explanations of particular topics.

For educators wishing to explore the use of LLMs in creating new learning experiences or developing "conversational" materials in a particular domain, being able to develop models locally can improve productivity and support innovation.

This note provides a living document to help track various approaches and proofs of concepts relating to the running and fine-tuning/domain specific training of LLMs, particularly on local devices.

## Toolchains for working with LLMs

So that we don't lock in to a particular LLM, ideally we want to work with an abstraction layer that allows us to plug in locally and remotely accessed LLMs as required, and as new models become available.

- `langchain` model toolchains provide a generic Python API to multiple models
https://langchain.readthedocs.io/en/latest/index.html

- `langflow` — graphical UI for `langchain` https://github.com/logspace-ai/langflow

- simple Python bindings for `llama.cpp`
  - https://github.com/abetlen/llama-cpp-python ([example langchain usage](https://github.com/abetlen/llama-cpp-python/blob/main/examples/langchain_custom_llm.py))
  - https://github.com/thomasantony/llamacpp-python
  - `fastLLaMa` https://github.com/PotatoSpudowski/fastLLaMa
  
 - `LocalAI` — "drop-in replacement API compatible with OpenAI for local CPU inferencing, based on llama.cpp, gpt4all, rwkv.cpp and ggml, including support GPT4ALL-J" https://github.com/go-skynet/LocalAI

## Running LLMs on local devices

The [`llama.cpp`](https://github.com/ggerganov/llama.cpp) port of the Facebook Llama model demonstrated how the model could be quantised to create models with a relatively small download size (4GB for the 7B model) that were capable of running on a home computer.

*See also [`lit-llama`](https://github.com/Lightning-AI/lit-llama), Apache2 licensed equivalent to `llama` (GPL).*

This bootstrapped a large number of projects based around the `llama.cpp` model:

- Stanford Alpaca — `llama.cpp` model trained against 52k QandA pairs generated from ChatGPT https://github.com/antimatter15/alpaca.cpp
- tutorials on training your own Alpaca style model:
  - https://simonwillison.net/2023/Mar/13/alpaca/
  - https://replicate.com/blog/replicate-alpaca
- reproducing the Stanford Alpaca results using low-rank adaptation (LoRA) https://github.com/tloen/alpaca-lora
- cleaned training data derived from original Stanford Alpaca training set https://github.com/gururise/AlpacaDataCleaned
- `Dalai` — simple app for running Stanford Alpaca locally https://cocktailpeanut.github.io/dalai/#/
- `alpaca.cpp` ("locally run an instruction-tuned chat-style LLM") — https://github.com/antimatter15/alpaca.cpp
- alpaca / WIP example of local LLM class for `langchain` — https://gist.github.com/lukestanley/6517823485f88a40a09979c1a19561ce
- alpaca in langchain from a HuggingFace model — https://medium.com/artificialis/crafting-an-engaging-chatbot-harnessing-the-power-of-alpaca-and-langchain-66a51cc9d6de
- alpaca in langchain using model from Huggingface https://m.youtube.com/watch?v=v6sF8Ed3nTE
- GPT4All - trained on ~800k GPT-3.5-Turbo generated prompts https://github.com/nomic-ai/gpt4all ; [example notebook / py client](https://github.com/nomic-ai/nomic/blob/main/examples/GPT4All.ipynb) — `nomic` package then `import nomic.gpt4all as gpt4all` and `gpt4all.prompt(p)`
- alpaca HF langchain conversation demo - https://gist.github.com/cedrickchee/9daedfff817e2b437012b3104bdbc5f3
- `langchain` alapac PR — https://github.com/hwchase17/langchain/pull/2297

## Models That Require a Commodity GPU / Running on Arbitrary Platforms

- https://github.com/Lightning-AI/lit-llama
- training https://github.com/bublint/ue5-llama-lora eg using https://github.com/oobabooga/text-generation-webui
- training https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama
- arbitray platfrom — " allows any language models to be deployed natively on a diverse set of hardware backends and native applications"  https://github.com/mlc-ai/mlc-llm

## Localised fine tunings

- Italian instruction tuned Llama model https://github.com/teelinsan/camoscio
- example of model traind on chat messages (incl. recipe) https://www.izzy.co/blogs/robo-boys.html
- replit code model (good human performance also?) [replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b)
- turbopilot — " self-hosted copilot clone which uses the library behind llama.cpp to run the 6 Billion Parameter Salesforce Codegen model in 4GiB of RAM" https://github.com/ravenscroftj/turbopilot

## Llama / Alpaca fine tuning

As well as the tutorials describing how to recreate the original Stanford Alpaca model, there's a growing number of examples of training using alternative training sets, eg for custom domains.

- speak in the style of Homer Simpson https://replicate.com/blog/fine-tune-llama-to-speak-like-homer-simpson

- simple UI to support fine tuning of Alpaca model https://github.com/lxe/simple-llama-finetuner

- issue — *How to finetune model with a new knowledge?* https://github.com/tloen/alpaca-lora/issues/45

- fine tuning against code https://github.com/sahil280114/codealpaca 

- lightweight adapter for fine-tuning instruction-following LLaMA models https://github.com/ZrrSkywalker/LLaMA-Adapter


## Training

- Replicate: https://replicate.com/docs/guides/fine-tune-a-language-model https://github.com/fofr/replicate-llm-training

## Alternatives to Llama

Whilst the original proof-of-concept "local running" models were based on the Facebook Llama model, examples of other fine-tuneable "DIY LLMs" are becoming available:

- Dolly / eleuther AI (open source model) https://github.com/databrickslabs/dolly

- BLOOM multingual LLM basis https://github.com/linhduongtuan/BLOOM-LORA

- RWKV (recurrent NN rather than transformers)
  - base model https://github.com/BlinkDL/RWKV-LM
  - chat version https://github.com/BlinkDL/ChatRWKV

- StableLM https://github.com/Stability-AI/StableLM
- flan-alpaca https://github.com/declare-lab/flan-alpaca
- LaMini-LM https://github.com/mbzuai-nlp/LaMini-LM
- WizardLM https://github.com/nlpxucan/WizardLM
- ReplitLM / Replit LLM https://github.com/replit/ReplitLM/ https://huggingface.co/replit/replit-code-v1-3b

## Personalities

eg in GPT4All GUI, [`ParisNeo/PyAIPersonality`](https://github.com/ParisNeo/PyAIPersonality) etc

## API

OpenAI compatible API wrapper for local LLMs https://github.com/go-skynet/LocalAI

## Generating Embeddings

Fine-tunings try to embed knowledge in the model to give a sense of recall. An alternative way of trying to reduce hallucinations is to retrieve content from a knowledge source that is likely to answer a question, and then use that to augment the prompt.

LLMs can be used to support semantic search by creating embeddings for a source text and allowing a look-up from an embedding generated for a search query. *This pattern is increasingly be wrapped in higher level, more abstracted patterns, APIs, etc.*

- *How to implement Q&A against your documentation with GPT3, embeddings and Datasette* https://simonwillison.net/2023/Jan/13/semantic-search-answers/
- useful embeddings (scikitlearn pipelines for various embedding generators)
https://github.com/koaning/embetter  
- example of generating embeddings with llama https://github.com/DiegoMoralesRoman/pyllamma.cpp/blob/master/test.py
- index unstructured data with foundation models
https://github.com/hazyresearch/meerkat
- sentence transformers - semantic search in py
https://www.sbert.net/index.html

In passing, this looks like a handy SQLite plugin for dropping in vector-based similarity search: [sqlitevss](https://github.com/asg017/sqlite-vss) [[about](https://observablehq.com/@asg017/introducing-sqlite-vss)]

## Memorising transformers

Memorising transformers could offer an interesting way of injecting additional knowledge into a conversational chain:

- *How To Scale Transformers’ Memory up to 262K Tokens* https://pub.towardsai.net/extending-transformers-by-memorizing-up-to-262k-tokens-f9e066108777
- implementation of Memorizing Transformers in Pytorch https://github.com/lucidrains/memorizing-transformers-pytorch

## Locally Run UIs

Many LLMs are capable of generating code or other text based scripts. Currently (March, 2023), these are presented as text (potentially, syntax highlighted text) in a simple text based UI. However, the code typically cannot be executed and its outputs displayed, nor rendered (eg in the case of text based diagram descriptions using mermaid.js or Graphviz dot syntax), inline.

For conversational use alongside a fixed text, or a conversation with a set of pre-existing materials that provide a knowledge base (for example, a text book or a set of learning course materials), what sort of UI would be most natural?

For other modalities (e.g. image generation), what sort of emerging UIs are available for accessing LLMs and other generative models?

- llama / Alpaca UIs
  - `Dalai` — simple app for running Stanford Alpaca locally https://cocktailpeanut.github.io/dalai/#/
  - user-friendly web UI for the alpaca.cpp language — https://github.com/ViperX7/Alpaca-Turbo
  - text generation web UI (gradio web UI for accessing a wide range of models, incl. llama) https://github.com/oobabooga/text-generation-webui

- Jupyter-ai (JupyterLab extension providing an API to geenrative AI services)
https://github.com/jupyterlab/jupyter-ai

- ChatGPT error handler for Jupyter https://github.com/hack-r/ChatGPT_for_Jupyter

- `TextSQL` — natural language queries, e.g. over US population dataset
https://github.com/caesarHQ/textSQL

- ask questions over a Notion database using natural language (based on `langchain`) https://github.com/hwchase17/notion-qa *TO DO: try this out with some OpenLearn materials...*

- "personify" a book to have a conversation with it https://github.com/batmanscode/Talk2Book ; (this should be updated to use the new [`langchain` *Retrieval* chain]) (https://blog.langchain.dev/retrieval/).

- GPT-4 & LangChain — ChatGPT Chatbot for Large PDF Files https://github.com/mayooear/gpt4-pdf-chatbot-langchain

- GPT4 assisted coding environment
https://www.cursor.so/

- Stable Diffusion locally served web-ui
https://github.com/AUTOMATIC1111/stable-diffusion-webui

- Coauthor interface https://github.com/minalee-research/coauthor-interface

- Question extractor "extract question/answer pairs automatically from existing textual data" ?can we do the same with a local LLM, not ChatGPT? https://github.com/nestordemeure/question_extractor 

See also: *ReAct-Based Patterns*

## ReAct-Based Patterns

The [*ReAct pattern*](https://react-lm.github.io/) interleaves text generation with the ability to act on generated texts, as well as using action outputs as further inputs to the conversation.

- examples https://interconnected.org/home/2023/03/16/singularity
- simple Python API for adding Wikipedia lookups, calculator execution, and search against a Datasette/SQLite database to OpenAI LLM conversations: https://til.simonwillison.net/llms/python-react-pattern
- example connectors for scraping websites, local document sources (PDF, Powerpoint, docx, Youtube video audio etc) to act as knowledge source in chatgpt prompt https://github.com/geeks-of-data/knowledge-gpt ([issue relating to finding open source LLM alternatives](https://github.com/geeks-of-data/knowledge-gpt/issues/64))

Note that `langchain` provides hooks for a wide range of *agents*, including search, a Python REPL, etc: https://python.langchain.com/en/latest/modules/agents.html

It's quite easy to hook in custom defined agents into `langchain`. For example:

- chatting about baseball stats: https://gist.github.com/geoffreylitt/b345e5a3fcc18368df04b49f6924c217 [[about](https://www.geoffreylitt.com/2023/01/29/fun-with-compositional-llms-querying-basketball-stats-with-gpt-3-statmuse-langchain.html)]

Recently announced [ChatGPT plugins](https://openai.com/blog/chatgpt-plugins) [[docs](https://platform.openai.com/docs/plugins/introduction)] demonstrate this pattern. The 'langchain` tooling can also hook into ChatGPT plugins: https://python.langchain.com/en/latest/modules/agents/tools/examples/chatgpt_plugins.html 

BabyAGI4all - small autonomous AI agent based on [BabyAGI](https://github.com/yoheinakajima/babyagi)  that runs with local LLMs — https://github.com/kroll-software/babyagi4all

## Question and Answer / QandA / QnA Patterns

PaperQA — "question and answering from PDFs or text files" https://github.com/whitead/paper-qa

Simple QnA using ggml-vicuna-7b-1.1-q4_2 https://gist.github.com/kroll-software/94328628cb069965dc34c7e594f50d09

## Plugin frameworks

Following the release of [ChatGPT plugins](https://platform.openai.com/docs/plugins/introduction), to what extent might we be able to create plugin frameworks for other models, and to what extent might we be able to create a generic plugin framework that could be applied to any LLM in similar way to how tools such as `langchain` provide an abstraction layer over calling different LLMs.

- example ChatGPT plugin that loops queries through datasette/sqlite https://simonwillison.net/2023/Mar/24/datasette-chatgpt-plugin/
- Llama retrieval plugin POC https://github.com/lastmile-ai/llama-retrieval-plugin [[about](https://blog.lastmileai.dev/using-openais-retrieval-plugin-with-llama-d2e0b6732f14)]

## Evaluation

How can we test or evaluate LLMs?

- Evaluating language chains https://blog.langchain.dev/evaluation/
- metric to test LLM's for progress in eliminating hallucinations https://github.com/manyoso/haltt4llm

## Browser Based Generative Models

There are various demonstrations of running LLMs and other generative systems purely in a browser. To run models in the browser, two implementation approaches are possible: Javascript based models and WASM models.

The WASM approach also supports the running of models in server or locally hosted containers. These containers are much lighter than "full fat" Docker containers, but can also be run using Docker machinery (eg [Docker + WASM announcement](https://www.docker.com/blog/docker-wasm-technical-preview/) and [Docker+Wasm Technical Preview 2](https://www.docker.com/blog/announcing-dockerwasm-technical-preview-2/)).

vicuna-7b LLm in a browser: https://github.com/mlc-ai/web-llm

<1 GB model running in a browser with no web GPU requirement?! https://github.com/lxe/wasm-gpt/tree/wasm-demo

### Javascript models

- Transformers.js - transformers in the browser
https://xenova.github.io/transformers.js/

### WASM-based Generative Models

Examples of WASM based generative models (text, images, etc).

- Stable diffusion https://github.com/mlc-ai/web-stable-diffusion  
- Llama https://github.com/ggerganov/llama.cpp/issues/97

## Large Multimodal Models (LMM)

OpenFlamingo https://laion.ai/blog/open-flamingo/

## Training From Scratch

MinimalGPT — " aconcise, adaptable, and streamlined code framework that encompasses the essential components necessary for the construction, training, inference, and fine-tuning of the GPT model" https://github.com/abhaskumarsinha/MinimalGPT/

