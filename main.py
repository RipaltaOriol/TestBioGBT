from transformers import AutoTokenizer, AutoModelForCausalLM


def generate():
    model_name = "microsoft/BioGPT-Large"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = "Genes that express proteins found in lysosomes include"

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(input_ids, max_length=500, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(generated_text)


def main():
    generate()


if __name__ == "__main__":
    main()