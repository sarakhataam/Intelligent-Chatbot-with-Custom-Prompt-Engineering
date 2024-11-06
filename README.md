# Intelligent-Chatbot-with-Custom-Prompt-Engineering
Here’s a version of the README written in the format of a README file:

---

# Healthcare Chatbot using DialoGPT

## Overview
This project involves building a conversational healthcare assistant using **DialoGPT** to provide friendly, supportive, and general health advice to users. The chatbot offers practical tips on common health issues and encourages consulting healthcare professionals for specific concerns. It was trained on custom doctor-patient dialogue data to simulate realistic, helpful conversations while maintaining a conversational and empathetic tone.

## Features
- **Natural Language Processing (NLP)**: Utilizes DialoGPT, a large-scale generative language model, fine-tuned on doctor-patient dialogue data.
- **Custom Dataset**: The model is trained using a dataset of medical interactions, offering responses tailored to healthcare-related queries.
- **Contextual Responses**: The chatbot maintains context within a conversation, making the interaction feel natural and relevant.
- **Preprocessing Techniques**: Data preprocessing includes handling contractions, lemmatization, tokenization, and stop word removal to improve model training and response quality.
- **MLflow Integration**: Tracks experiments and logs model performance metrics for easy monitoring and reproducibility.
- **Interactive Interface**: Allows users to input health-related questions and receive immediate responses.

## Project Highlights
- **Preprocessing**: Cleaned and tokenized text data, expanded contractions, removed stop words, and lemmatized words for efficient processing.
- **Model Training**: Fine-tuned **DialoGPT-medium** using PyTorch, optimizing for conversational healthcare responses.
- **Evaluation**: Employed a custom dataset split into training and testing sets, evaluating the model's performance and generating diverse responses.
- **Response Generation**: Implemented response generation with customizable parameters such as temperature, top_k, and top_p to control creativity and randomness in responses.

## Setup Instructions

### Prerequisites
- Python 3.7+
- PyTorch
- Hugging Face's Transformers library
- MLflow for experiment tracking
- NLTK for text preprocessing
- Contractions and Num2Words libraries for text normalization
- Other dependencies are listed in `requirements.txt`

### Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/healthcare-chatbot.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Download and set up **DialoGPT**:
    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    ```

### Training the Model
1. Preprocess the data:
   - Data is processed by converting text to lowercase, removing numbers, and expanding contractions.
   - Tokenization and lemmatization steps are applied to clean the text.

2. Train the chatbot:
   - The model is trained using the **Trainer** API with custom `TrainingArguments`.
   - The chatbot was trained for **15 epochs** with a batch size of **2**, and the trained model is saved for future use.

### Generating Responses
To generate a response from the chatbot, use the following code:

```python
response = generate_response("What are the possible causes of lower back pain?")
print("Chatbot Response:", response)
```

### Logging with MLflow
The project integrates MLflow to log model parameters, responses, and performance metrics. Start an MLflow run before training:

```python
mlflow.set_experiment("Chatbot_Model_Experiment")
with mlflow.start_run():
    trainer.train()
```

## Usage
Run the chatbot in interactive mode:

```bash
python chatbot.py
```

The chatbot will prompt you to ask health-related questions. Type your query, and it will respond with advice based on the conversation history.

## Example Conversation
```
User: I have a headache and feeling tired.
Chatbot: I'm sorry to hear that. Make sure to stay hydrated and rest. If your headache persists, consider reaching out to a healthcare provider for advice.

User: What are the possible causes of lower back pain?
Chatbot: Lower back pain can be caused by muscle strain, poor posture, or long periods of sitting. If the pain continues, it’s best to consult a doctor for further evaluation.
```

## Future Improvements
- **Model Expansion**: Fine-tune the model on more diverse datasets, including mental health conversations or specialized medical topics.
- **Enhanced Conversational Flow**: Improve the model’s ability to manage longer, multi-turn conversations with better context retention.
- **Integration**: Connect the chatbot with a web or mobile app interface for broader accessibility.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

This README file provides a clear and professional overview of your project, guiding users on how to install, train, and use the healthcare chatbot.
