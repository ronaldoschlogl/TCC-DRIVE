//rodar
python treinamento_difusao.py --load ./saved_models/model_20000.pth

python teste.py --num_sample 250 --diffusion_model ./saved_models/model_20000.pth
