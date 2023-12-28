from codegeex.tokenizer import CodeGeeXTokenizer

tokenizer = CodeGeeXTokenizer(tokenizer_path='codegeex/tokenizer', mode='codegeex-13b')
test_prompt = "def test():\n    print('hello world')\n    return 0"

input_ids = tokenizer.encode_code(test_prompt)
print(tokenizer.tokenizer.convert_ids_to_tokens(input_ids))

tokenizer.decode_code(input_ids)
