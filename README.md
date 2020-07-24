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
  * Inference (compose)
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
  * Inference w/ ``inference.py``

## Structure
```
├── data_preprocess.sh      (executes python scripts to build vocab and prepare data) 
├── inference.py            (generates Jazz music)
├── requirements.txt        (python dependencies)
├── train.py                (trains Transformer-XL from scratch)
├── data                    (.pkl files for training)
├── mcsv_beat               (Jazzomat dataset)
├── mcsv_melody             (Jazzomat dataset)
├── output                  (sample generated piece)
│   ├── demo.csv
│   ├── demo.midi
├── pickles                 (houses required metadata for training)
├── remi_encs_struct        (contains training data in readable REMI event sequences)
├── src                     (codes for preprocessing)
│   ├── build_chord_profile.py   (reads and stores key templates for different chord types defined in ``chord_profile.txt``)
│   ├── build_vocab.py           (builds the vocabulary for the Jazz Transformer)
│   ├── chord_processor.py       (class and methods for converting notes to chords and vice versa)
│   ├── chord_profile.txt        (hand-crafted key templates for each chord type)
│   ├── containers.py            (container classes for events in mcsv files)
│   ├── convert_to_remi.py       (converts Jazzomat dataset to REMI events for training)
│   ├── mcsv_to_midi.py          (converts mcsv file to midi format)
│   ├── midi_decoder.py          (class and methods for conversion from REMI to midi)
│   ├── mlu_processor.py         (class and methods for defining and parsing Mid-level Unit (MLU) events)
│   ├── remi_containers.py       (container classes for REMI events)
│   ├── utils.py                 (miscellaneous utilities)
├── transformer_xl          (codes for the model)
│   ├── model_aug.py             (Jazz Transformer model)
│   ├── modules.py               (functions for constructing Transformer-XL)
```

