import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from codecarbon import OfflineEmissionsTracker

##################FUNCTIONS#######################

#Load configurations in a list of dictionaries
def carica_tutte_le_righe(filename):
    data=[]
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Pulisce la riga e divide i valori
            values = line.split()
            # Verifica che ci siano abbastanza valori nella riga
            if len(values) == 4:
                # Crea un dizionario per ogni riga e aggiunge alla lista dati
                data.append({
                    'max_length': int(float(values[0])),
                    'top_p': float(values[1]),
                    'top_k': int(float(values[2])),
                    'temperature': float(values[3]),
                    'do_sample': True
                })
            else:
                print(f"Riga non valida: {line}")
    return data

# Function to get the config from dict list
def get_conf(riga_index):
    if riga_index <= len(dati):
        return dati[riga_index - 1]
    else:
        raise IndexError("La riga richiesta non esiste nel file.")

#################################################

base_path = 'leonardo/mistral/few-shots'
examples_base_path = 'examples/few-shots'
device = 'cuda'
model_id = 'models/Mistral-7B-Instruct-v0.2'
file_name = 'hyperparametersConfig.txt'
gpu_ids = [0,1,2,3]

#Create the parser
parser = argparse.ArgumentParser(description='How many inference cycles')

#Add arguments
parser.add_argument('first', type=int, help='Start value')

#Parsing of the arguments
args = parser.parse_args()

#Load the model and the tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = 'left')
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map = "auto"
    )
    
#Load the prompts in a structure
instructions = ["Imagine you are in an imaginary world. Describe the fantastic setting and the adventures that can be experienced.", "Imagine being in a fictional world inhabited by both humans and mythical creatures. Describe the fantastic setting and the adventures that can be experienced. Make sure to capture all of this.", "Imagine being in a fictional world inhabited by both humans and mythical creatures. Describe the fantastical setting, including enchanted landscapes and mysterious places, and the adventures that can be experienced. Make sure to capture all of this.","Imagine being in a fictional world inhabited by both humans and mythical creatures. Describe the fantastical setting, including enchanted landscapes, ancestral forests, and snow-capped mountains, and the adventures that can be experienced. Make sure to capture all of this.", "Imagine being in a fictional world inhabited by both humans and mythical creatures. Describe the fantastical setting, including enchanted landscapes, ancestral forests populated by talking trees and magical creatures, and the adventures that can be experienced. Make sure to capture all of this.",
         "Imagine being in a fictional world inhabited by both humans and mythical creatures. Describe the fantastical setting, including enchanted landscapes, ancestral forests populated by talking trees and magical creatures, and the adventures that can be experienced, such as searching for ancient hidden treasures or fighting against dark masters of evil. Make sure to capture all of this.", "Imagine being in a fictional world inhabited by both humans and mythical creatures. Describe the fantastical setting, including enchanted landscapes, ancestral forests populated by talking trees and magical creatures, mysterious cities and haunted castles, and the adventures that can be experienced, such as exploring underground worlds or fighting giant dragons. Make sure to capture all of this.",
         "Imagine being in a fictional world inhabited by both humans and mythical creatures. Describe the fantastical setting, including enchanted landscapes, ancestral forests populated by talking trees and magical creatures, mysterious cities and haunted castles, stormy seas and remote islands, and the adventures that can be experienced, such as finding the source of eternal life or defending the kingdom against an army of the undead. Make sure to capture all of this.",
         "Imagine being in a fictional world inhabited by both humans and mythical creatures. Describe the fantastical setting, including enchanted landscapes, ancestral forests populated by talking trees and magical creatures, mysterious cities and haunted castles, stormy seas and remote islands, fiery deserts and frozen lands, and the adventures that can be experienced, such as the search for the divine artifact or the liberation of an entire enslaved race. Make sure to capture all of this.",
         "Imagine being in a fictional world inhabited by both humans and mythical creatures. Describe the fantastical setting, including enchanted landscapes, ancestral forests populated by talking trees and magical creatures, mysterious cities and haunted castles, stormy seas and remote islands, fiery deserts and frozen lands, cloud-touching mountains and endless plains, and the adventures that can be experienced, such as the quest for the book of forbidden spells or the final battle against the ultimate evil. Make sure to capture all of this."]

#Execute inferences and track emissions
x = args.first
data = carica_tutte_le_righe(file_name)
config = get_conf(x)
#Create a folder and save outputs
output_dir=os.path.join(base_path, str(x))
os.makedirs(output_dir, exist_ok=True)
#Execute inference
output_file_path = os.path.join(output_dir, 'output.txt')
with open(output_file_path, 'w', encoding='utf-8') as file:
    for key, value in config.items():
        file.write(f"{key}: {value}\n")
    for idx, instruction in enumerate(instructions, start=1):
        #Example path
        example_path = os.path.join(examples_base_path, f'{str(x)}.txt') 
        # Leggi il contenuto del file di esempio
        with open(example_path, 'r', encoding='utf-8') as example_file:
            example_content = example_file.read()
        #Concatenate the example to the prompt
        concatenated_instruction = f'{instruction}\n{example_content}'
        tracker = OfflineEmissionsTracker(project_name="Mistral7BInstruct0.2_fewShot_leonardo", gpu_ids=gpu_ids,tracking_mode = 'process', country_iso_code="ITA", output_dir=output_dir)
        messages = [
            { "role" : "user", "content" : concatenated_instruction }
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
            
        #Compute the prompt length
        prompt_length = inputs.size(1)

        #Get the max_length value from the config 
        original_max_length = config.get("max_length", 1024)

        # Adjust max_length counting the prompt length
        adjusted_max_length = original_max_length + prompt_length

        # Remove max_length from config to avoid conflicts
        config = {k: v for k, v in config.items() if k != 'max_length'}
        tracker.start()
        try:
            generated_ids = model.generate(inputs,max_length=adjusted_max_length, **config, pad_token_id=tokenizer.eos_token_id)
        finally:
            tracker.stop()
            decoded = tokenizer.batch_decode(generated_ids)
            file.write(decoded[0])
