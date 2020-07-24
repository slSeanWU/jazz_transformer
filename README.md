# The Jazz Transformer

An adapted Transformer-XL deep learning model that composes Jazz music  (lead sheets&mdash;_chord progression & melody_).

Tensorflow implementation of the automatic music composition model presented in our paper: 
 * **The Jazz Transformer on the Front Line: Exploring the Shortcomings of AI-composed Music through Quantitative Measures**  
   Shih-Lun Wu and Yi-Hsuan Yang  
   (To appear at) _The 21st International Society for Music Information Retrieval Conference_ (ISMIR), 2020.

## Usage Notes
### Prerequisites
 * **Python 3.6** ([install](https://www.python.org/downloads/release/python-368/))
 * _Recommended_: a working GPU with &geq;2GB of memory
 * Install dependencies (``pip`` or ``pip3``, depending on your sytem)
  ```shell
  pip3 install -r requirements.txt
  ```
  
### Compose Some Songs Right Away
  * Download pretrained model
  ```shell
  [command to get the checkpoint]
  ```
  * Inference with ``inference.py``
  ```shell
  python3 inference.py [--model MODEL] [--temp TEMP] [--csv CSV] output_filename
  ```
  ``output_filename``:  output midi file path  
  ``--model MODEL``:    path to the trained model checkpoint (default: the downloaded model)  
  ``--temp TEMP``:      sampling temperature for generation (default: ``1.2``)  
  ``--csv CSV ``:       (optional) output csv file path (which records the generated event sequence)  

### Training from Scratch
  * Preprocess dataset
  ```
  ./data_preprocess.sh
  ```
  * Train the model
  ```
  python3 train.py checkpoint_filepath log_file
  ```
  ``checkpoint_filepath``:  the folder to save checkpoints  
  ``log_file``:             the file path to save log file  

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

