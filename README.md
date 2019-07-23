# Large-Scale Transfer Learning for Natural Language Generation

### Team

* Sergey Golovanov - sergey.golovanov@neuromation.io
* Rauf Kurbanov - rauf.kurbanov@neuromation.io
* Sergey Nikolenko - snikolenko@neuromation.io
* Kyryl Truskovskyi - kyryl@neuromation.io
* Alexander Tselousov - al.tselousov@gmail.com
* Thomas Wolf - thomas@huggingface.co

### How to run

Unzip BPE vocabulary files into `./parametes` folder and save checkpoint into 
`./checkpoints` folder or use scripts (see below). 

The easiest way to prepare environment is to run script `prepare_environment.sh`.
After that docker container with retrieval server must be run in demon mode and 
image with `transformer_chatbot` must be built. Run scripts from the root folder of this repository.

After preparations metrics can be evaluated with corresponding `docker_*.sh` scripts or
`*.py` scripts can be used during interactive container run. 

List of used python modules is in `requirements.txt`. Also `pytorch=1.0` is used.

# Run all experiments on the platform

All experiments were run on Neuromation platform

```
./platform/train.sh

```
