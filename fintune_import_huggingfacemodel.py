from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Your model name on Hugging Face Hub
model_name = "ChandrikaManchikanti/finetuned-mbart-slang-literal-en-hi"

# Load tokenizer and model
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Set the source and target languages
tokenizer.src_lang = "en_XX"
target_lang = "hi_IN"

def translate(text):
    # Tokenize input with forced language code prefix
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Generate translation
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
    )

    # Decode the output tokens
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation
# Literal context
print(translate("literal:The soup tasted salty."))
print(translate("literal:His car crashed into a tree."))
print(translate("literal:He grabbed the clutch quickly."))
print(translate("literal:He spilled water on the documents."))
print(translate("literal:The thief tried to steal the phone."))
print(translate("literal:The boy slipped and hit his head."))

# Slang context
print(translate("slang:I'm not even mad, that's impressive."))
print(translate("slang:That hit different."))
print(translate("slang:That’s cringe."))
print(translate("slang:That deal was a steal!"))
print(translate("slang:Quit playing, spill the tea!"))
print(translate("slang:Let’s chill out."))
print(translate("slang:That's dope"))
