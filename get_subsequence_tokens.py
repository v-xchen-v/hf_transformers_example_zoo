from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

# Here is an example sentence, constructed by context followed by continuation.
context = "Aubrey took tennis lessons as a method to get in shape.\nQuestion: What does Aubrey need to do before this?\nAnswer:"
continuation = "get tennis clothes"
# continuous = " A"
sequence = f"{context}{continuation}"

# Tokenize the sentence, context, and continuation texts, or obtain continuation token from the correct slice of sequence tokens.
seq_tokens = tokenizer.tokenize(sequence)
print("sequence tokens:", seq_tokens)
# sequence tokens: ['▁A', 'ubre', 'y', '▁took', '▁tennis', '▁less', 'ons', '▁as', '▁a', '▁method', '▁to', '▁get', '▁in', '▁shape', '.', '<0x0A>', 'Question', ':', '▁What', '▁does', '▁A', 'ubre', 'y', '▁need', '▁to', '▁do', '▁before', '▁this', '?', '<0x0A>', 'Answer', ':', '▁A', '▁A']

ctx_tokens = tokenizer.tokenize(context)
print("context tokens:", ctx_tokens)
# context tokens: ['▁A', 'ubre', 'y', '▁took', '▁tennis', '▁less', 'ons', '▁as', '▁a', '▁method', '▁to', '▁get', '▁in', '▁shape', '.', '<0x0A>', 'Question', ':', '▁What', '▁does', '▁A', 'ubre', 'y', '▁need', '▁to', '▁do', '▁before', '▁this', '?', '<0x0A>', 'Answer', ':']

continuation_tokens = tokenizer.tokenize(continuation)
print("continuation_tokens:", continuation_tokens)
print("continuation_tokens as IDs:", tokenizer.convert_tokens_to_ids(continuation_tokens))
continuation_decoded = tokenizer.convert_tokens_to_string(continuation_tokens)
print(continuation_decoded)
# continuation_tokens: ['▁', '▁get', '▁tennis', '▁clothes']
# continuation_tokens as IDs: [679, 22556, 22095]
#  get tennis clothes

continuation_as_subsequence_tokens = seq_tokens[len(tokenizer.tokenize(context)):]
print("continuation_as_subsequence_tokens:", continuation_as_subsequence_tokens)
print("continuation_as_subsequence_tokens as IDs:", tokenizer.convert_tokens_to_ids(continuation_as_subsequence_tokens))
decoded_continuation_as_subsequence = tokenizer.convert_tokens_to_string(continuation_as_subsequence_tokens)
print(decoded_continuation_as_subsequence)
# continuation_as_subsequence_tokens: ['▁get', '▁tennis', '▁clothes']
# continuation_as_subsequence_tokens as IDs: [657, 22556, 22095]
# get tennis clothes

# if continuation.startswith(" ") or not
assert ctx_tokens + continuation_as_subsequence_tokens == seq_tokens
assert ctx_tokens + continuation_tokens != seq_tokens
# If we want to get continuation as subsequence we should obtain from correct slice of sequence token instead of tokenizing directly which matter when evaluating generated continuation from LLM models.