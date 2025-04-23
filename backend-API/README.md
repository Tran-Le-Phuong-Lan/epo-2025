# How to run the file `semantic-search_lan.py`?

## Install the followings

[1] ! pip install datasets transformers faiss-cpu torch llama-cpp-python "fastapi[standard]"

[2] your computer RAM >= 8 GB

## Download data, models for local usage

[1] Download sample of preprocessed [database](https://drive.google.com/drive/folders/18eT4cgdDBvNBz8JhS_vd-eZnS9R8S1wM?usp=drive_link) into the **same directory with the file `semantic-search_lan.py`**

[2] Download model for generating embeddings [arnab_model](https://drive.google.com/drive/folders/1YvhT3xINPBepMGUsw5ydrvzM4D6KgIMA?usp=drive_link) 

[3] Download model for generating text [mistral_7b_instruct_v0.3_gguf](https://drive.google.com/drive/folders/1UVoiIvQEdhqZ13OUSWyA51_NR-47q9Z_?usp=drive_link)

## Run and test `semantic-search_lan.py`

[1] Run the file: in `git bash`, run the command `fastapi dev semantic-search_lan.py`

[2] Test it using `postman` to fake client to send `POST` request (ask **MAU**). The test protocal (due to code drawback, will be updated when necessary) must follow the order

    [2.0] Must modify the path variabls in `semantic-search_lan.py`:

        `model_ckpt` modified to point to the folder downloaded from **arnab_model** link

        `model_pre_downloaded_path` modified to point to file **Mistral-7B-Instruct-v0.3.Q4_K_M.gguf** downloaded from **mistral_7b_instruct_v0.3_gguf** link

    [2.1] send payload `{"query": "how to test nuclear acids in a sample?", "top_k": 5}` in the `POST` request to `/search`. The value of key `query` could be changed but to see how things works at the moment, you should at least know the `claims` of the **database** (section **Download data, models for local usage** above). The value of key `top_k` to specify the top number of `k` most relevant documents from the **database**

    [2.2] send payload `{"query": "how to test nuclear acids in a sample?", "top_k": 5}` in the `POST` request to `/answer`. **Only the value `query` matters**. You can specify whatever question you want as value for the key `query`, You can send as many questions as you want with the retrieved documents from [2.1]. However, if you want to change the context (i.e the retreived documents, repeat from [2.1]) 



