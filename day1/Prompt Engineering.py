from llmlingua import PromptCompressor

# llm_lingua = PromptCompressor(
#     model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
#     use_llmlingua2=True, # Whether to use llmlingua-2
# )

## Use LLMLingua-2-small model
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True, # Whether to use llmlingua-2
)

prompt = """ChatGPT is a generative artificial intelligence (AI) chatbot[2][3] developed by OpenAI and launched in 2022. It is based on the GPT-4o large language model (LLM). ChatGPT can generate human-like conversational responses, and enables users to refine and steer a conversation towards a desired length, format, style, level of detail, and language.[4] It is credited with accelerating the AI boom, which has led to ongoing rapid investment in and public attention to the field of artificial intelligence.[5] Some observers have raised concern about the potential of ChatGPT and similar programs to displace human intelligence, enable plagiarism, or fuel misinformation."""

compressed_prompt = llm_lingua.compress_prompt(prompt, rate=0.33, force_tokens = ['\n', '?'])

print('original prompt length: ', len(prompt))
print('compressed prompt length: ',len(compressed_prompt['compressed_prompt']))
compressed_prompt['compressed_prompt']