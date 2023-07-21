import openai
import time

OPENAI_API_KEY = ''
MODEL_ENGINE = "text-davinci-003"
QUESTION = (
    "Can you please write this text in chronological order and make it more readable?"
)

# Set up the OpenAI API client
openai.api_key = OPENAI_API_KEY


def gen_query(prod_title, prod_review):
    part_1 = f'Given a product witht the title {prod_title}, can you rate how useful the review below is, on a scale from 1-5, where 1 is not useful, and 5 very useful? Please only respond with the number, no other text, and please translate both the title and the review in English if they are in another language'
    full_q = f'{part_1} \n Review: {prod_review}'
    return full_q

def get_chatgpt_score(prod_title, prod_review):
    # Generate a response from OpenAI
    time.sleep(2)
    prompt = gen_query(prod_title, prod_review.split(' ')[:100])
    completion = openai.Completion.create(
        engine=MODEL_ENGINE,
        prompt=prompt,
        max_tokens=10,
        n=1,
        stop=None,
        temperature=0.2,
    )

    response = completion.choices[0].text

    return response


