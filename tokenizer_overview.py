from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")


# TOKENIZATION
# the `tokenize()` method takes a sentence string as input, and the output is a list of list, or tokens which could be words, characateres or subwords according to the type of tokenizer.
# sequence = "Using a transformer network is simple"
sequence = "The following are multiple choice questions (with answers) about siqa.\n\nJesse lived with their wife to help with the bills and to be happy together.\nQuestion: Why did Jesse do this?\nA. great\nB. bored\nC. get married\nAnswer: A"
tokens = tokenizer.tokenize(sequence)

print(tokens)
# ['▁The', '▁following', '▁are', '▁multiple', '▁choice', '▁questions', '▁(', 'with', '▁answers', ')', '▁about', '▁si', 'qa', '.', '<0x0A>', '<0x0A>', 'J', 'esse', '▁lived', '▁with', '▁their', '▁wife', '▁to', '▁help', '▁with', '▁the', '▁b', 'ills', '▁and', '▁to', '▁be', '▁happy', '▁together', '.', '<0x0A>', 'Question', ':', '▁Why', '▁did', '▁J', 'esse', '▁do', '▁this', '?', '<0x0A>', 'A', '.', '▁great', '<0x0A>', 'B', '.', '▁b', 'ored', '<0x0A>', 'C', '.', '▁get', '▁married', '<0x0A>', 'Answer', ':', 'A']
# This tokenizer a subword tokenizer: it splits the words until it obtain tokens that can represented by its vocabulary. That the case here with 'Jesse' which is split into two tokens: 'J' and 'esse'.
# If the tokenizer is character tokenizer, the output will be a list of characters. If for word tokenizer, the output will be a list of words.


# FROM TOKENS TO INPUT IDS
# the conversion to input IDs is handled by `convert_tokens_to_ids()` method.
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
# [450, 1494, 526, 2999, 7348, 5155, 313, 2541, 6089, 29897, 1048, 1354, 25621, 29889, 13, 13, 29967, 5340, 10600, 411, 1009, 6532, 304, 1371, 411, 278, 289, 6090, 322, 304, 367, 9796, 4208, 29889, 13, 16492, 29901, 3750, 1258, 435, 5340, 437, 445, 29973, 13, 29909, 29889, 2107, 13, 29933, 29889, 289, 4395, 13, 29907, 29889, 679, 8300, 13, 22550, 29901, 319]
# This outputs, once converted to the appropriate framework tensor, can then be used as model input.


# DECODE
# Decoding is going on the other way around: from vocabulary indices, we want to get a string. That can be done with the decode() as follow:
decoded_string = tokenizer.decode(ids)
print(decoded_string)

# Also, we can splits `decode()` into two steps: `converted_ids_to_tokens()` then followed by `convert_tokens_to_string()` that firstly converts the indices back to tokens, then groups together the tokens that were part of the same words to produce a readable sentence.
# ids = [319, 29909, 319, 319] # token_to_ids = { '_A': 319, 'A': 29909 }
tokens = tokenizer.convert_ids_to_tokens(ids)
print(tokens)
print(tokenizer.convert_tokens_to_string(tokens))