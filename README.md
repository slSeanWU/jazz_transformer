# The Jazz Transformer

An adapted Transformer-XL deep learning model that composes Jazz music  (lead sheets&mdash;_chord progression & melody_).

Tensorflow implementation of the automatic music composition model presented in our paper: 
 * **The Jazz Transformer on the Front Line: Exploring the Shortcomings of AI-composed Music through Quantitative Measures**  
   Shih-Lun Wu and Yi-Hsuan Yang  
   _The 21st International Society for Music Information Retrieval Conference_ (ISMIR), 2020.
   
_Want to listen to some compositions by the Jazz Transformer first?_ Click [here](https://drive.google.com/drive/folders/1-09SoxumYPdYetsUWHIHSugK99E2tNYD?usp=sharing)!

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
  ./download_model.sh
  ```
  * Inference (compose)
  ```shell
  python3 inference.py [--model MODEL] [--temp TEMP] [--struct_csv CSV] [--n_bars N_BARS] output_midi
  ```
  * ``output_midi``:  path to the output MIDI file
  * ``--model MODEL``:    path to the trained model checkpoint (default: the downloaded checkpoint)  
  * ``--temp TEMP``:      sampling temperature for generation (default: ``1.2``)
  * ``--n_bars N_BARS``:      # of bars to generate (default: ``32``)  
  * ``--struct_csv CSV ``:       path to the output csv file that records generated structure-related events (optional)

### Train from Scratch
  * Preprocess dataset
  ```
  ./data_preprocess.sh
  ```
  * Train the model
  ```
  python3 train.py ckpt_dir log_file
  ```
  * ``ckpt_dir``:  directory to save checkpoints  
  * ``log_file``:             path to the log file  
  
  Likewise, you may compose music with the model trained by yourself using ``inference.py`` (see above for instructions)

## Directory Structure
```
├── data_preprocess.sh      (executes python scripts to build vocab and prepare data) 
├── inference.py            (generates Jazz music)
├── requirements.txt        (python dependencies)
├── train.py                (trains Transformer-XL from scratch)
├── data                    (.pkl files for training)
├── mcsv_beat               (Jazzomat dataset content---beats+chords)
├── mcsv_melody             (Jazzomat dataset content---solo melody)
├── output                  (sample generated piece)
│   ├── demo.csv
│   ├── demo.midi
├── pickles                 (houses required metadata for training)
├── remi_encs_struct        (contains training data in readable REMI event sequences)
├── src
│   ├── build_chord_profile.py   (reads and stores key templates for different chord types defined in ``chord_profile.txt``)
│   ├── build_vocab.py           (builds the vocabulary for the Jazz Transformer)
│   ├── chord_processor.py       (the class and methods for converting notes to chords and vice versa)
│   ├── chord_profile.txt        (hand-crafted key templates for each chord type)
│   ├── containers.py            (container classes for events in mcsv files)
│   ├── convert_to_remi.py       (converts Jazzomat dataset to REMI events for training)
│   ├── explore_mcsv.py          (utilities for reading events from dataset .csv files)
│   ├── mcsv_to_midi.py          (converts mcsv file to midi format)
│   ├── midi_decoder.py          (the class and methods for conversion from REMI to midi)
│   ├── mlus_events.txt          (the mlu events used by the Jazz Transformer)
│   ├── mlu_processor.py         (the class and methods for defining and parsing Mid-level Unit (MLU) events)
│   ├── prepare_data.py          (splits data into training and validation sets before training the Jazz transformer)
│   ├── remi_containers.py       (container classes for REMI events)
│   ├── utils.py                 (miscellaneous utilities)
├── transformer_xl
│   ├── model_aug.py             (Jazz Transformer model)
│   ├── modules.py               (functions for constructing Transformer-XL)
```

## Acknowledgements
The Jazz Transformer is trained on the Weimar Jazz Database (**WJazzD**), a dataset meticulously annotated by the **Jazzomat Research Project** (_@ University of Music FRANZ LISZT Weimar_). Many thanks to them for the great work and making it publicly accessible!
   * URL for **WJazzD**: https://jazzomat.hfm-weimar.de/dbformat/dboverview.html
   
Also, we would like to thank **Yi-Jen Shih** (_@ NTUEE_, [personal GitHub](https://github.com/atosystem)) for the help he provided in arranging the codes of this repository.

## See Also
* The repository of evaluation metrics (also proposed in our paper) for machine-composed music:  
  https://github.com/slSeanWU/MusDr
