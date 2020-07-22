# Jazz Transformer

## Usage

### Prerequisite

```
requirements.txt
```
### data preprocess
```
$ ./data_preprocess.sh
```

### get pretrain model for inference
```
$ ./get_pretrain_model.sh
```

### Training

```
python train.py checkpoint_filepath log_file
```
* checkpoint_filepath  the folder to save checkpoints
* log_file             the file path to save log file


### Inference
xl_inference.py
```
python inference.py [--model MODEL] [--temp TEMP] [--csv CSV] output_filename
```
* output_filename  the output midi file path
* --model MODEL    model name for inference default : a pretrain model with loss 0.214
* --temp TEMP      temperature for inference (default 1.2)
* --csv CSV        (optional) output csv file path

## Structure
```bash
├── .gitignore
├── data_preprocess.sh
├── get_pretrain_model.sh
├── inference.py
├── requirements.txt
├── train.py
├── data
├── mcsv_beat
├── mcsv_melody
├── output
│   ├── demo.csv
│   ├── demo.midi
├── pickles
├── pretrain_model
├── remi_encs_struct
├── src
│   ├── build_chord_profile.py
│   ├── build_vocab.py
│   ├── chord_processor.py
│   ├── chord_profile.txt
│   ├── containers.py
│   ├── convert_to_remi.py
│   ├── explore_mcsv.py
│   ├── mcsv_to_midi.py
│   ├── midi_decoder.py
│   ├── mlu_processor.py
│   ├── mlus_refined.tmp
│   ├── prepare_data.py
│   ├── remi_containers.py
│   ├── utils.py
├── transformer_xl
│   ├── model_aug.py
│   ├── modules.py
│   ├── utils.py
```

