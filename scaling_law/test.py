from be_great import GReaT
from sklearn.datasets import fetch_california_housing
import time


start_time = time.time()

data = fetch_california_housing(as_frame=True).frame

data.to_csv("../data/test/data.csv", index=False)

model = GReaT(llm='distilgpt2', batch_size=16, epochs=1, save_steps=400000)
model.fit(data)

model_save_path = "../model/california_llm"

model.save(path=model_save_path)

model.load_from_dir(path=model_save_path)


synthetic_data = model.sample(n_samples=10000)

synthetic_data.to_csv("../data/test/synthetic_data.csv", index=False)

print(synthetic_data)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken to run the whole script: {elapsed_time} seconds")


