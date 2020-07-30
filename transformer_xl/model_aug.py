import tensorflow as tf
import numpy as np
import modules
import utils
import time
import os
import random
from copy import deepcopy

class TransformerXL(object):
    ########################################
    # initialize
    ########################################
    def __init__(self, event2word, word2event, checkpoint=None, is_training=False, training_seqs=None):
        # load dictionary
        self.event2word = event2word
        self.word2event = word2event
        # model settings
        self.x_len = 512      #input sequence length
        self.mem_len = 512    #
        self.n_layer = 12
        self.d_embed = 512
        self.d_model = 512
        self.dropout = 0.1
        self.n_head = 8
        self.d_head = self.d_model // self.n_head
        self.d_ff = 2048
        self.n_token = len(self.event2word)
        self.learning_rate = 2e-4
        self.group_size = 3
        self.entry_len = self.group_size * self.x_len
        # mode
        self.is_training = is_training
        self.training_seqs = training_seqs
        self.checkpoint = checkpoint
        if self.is_training: # train from scratch or finetune
            self.batch_size = 8
        else: # inference
            self.batch_size = 1
        # load model
        self.load_model()

    ########################################
    # load model
    ########################################
    def load_model(self):
        tf.compat.v1.disable_eager_execution()
        # placeholders
        self.x = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.y = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.mems_i = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, self.batch_size, self.d_model]) for _ in range(self.n_layer)]
        # model
        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        # initialize parameters
        initializer = tf.compat.v1.initializers.random_normal(stddev=0.02, seed=None)
        proj_initializer = tf.compat.v1.initializers.random_normal(stddev=0.01, seed=None)
        
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            xx = tf.transpose(self.x, [1, 0])
            yy = tf.transpose(self.y, [1, 0])
            loss, self.logits, self.new_mem = modules.transformer(
                dec_inp=xx,
                target=yy,
                mems=self.mems_i,
                n_token=self.n_token,
                n_layer=self.n_layer,
                d_model=self.d_model,
                d_embed=self.d_embed,
                n_head=self.n_head,
                d_head=self.d_head,
                d_inner=self.d_ff,
                dropout=self.dropout,
                dropatt=self.dropout,
                initializer=initializer,
                proj_initializer=proj_initializer,
                is_training=self.is_training,
                mem_len=self.mem_len,
                cutoffs=[],
                div_val=-1,
                tie_projs=[],
                same_length=False,
                clamp_len=-1,
                input_perms=None,
                target_perms=None,
                head_target=None,
                untie_r=False,
                proj_same_dim=True)
        self.avg_loss = tf.reduce_mean(loss)
        # vars
        all_vars = tf.compat.v1.trainable_variables()
        print ('num parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in all_vars]))
        grads = tf.gradients(self.avg_loss, all_vars)
        grads_and_vars = list(zip(grads, all_vars))
        # gradient clipping
        grads_and_vars = [(tf.clip_by_norm(grad, 100.), var) for grad, var in grads_and_vars]
        all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.compat.v1.trainable_variables()])
        # optimizer
        decay_lr = tf.compat.v1.train.cosine_decay(
            self.learning_rate,
            global_step=self.global_step,
            decay_steps=300000,
            alpha=0.004)
        # decay_lr = tf.compat.v1.train.cosine_decay_warmup(
        #     self.learning_rate,
        #     global_step=self.global_step,
        #     decay_steps=400000,
        #     warmup_steps=200,
        #     alpha=0.004
        # )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=decay_lr)
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, self.global_step)
        # saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=100)
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        # load pre-trained checkpoint or note
        if self.checkpoint:
            self.saver.restore(self.sess, self.checkpoint)
        else:
            self.sess.run(tf.compat.v1.global_variables_initializer())

    ########################################
    # data augmentation
    ########################################
    # return 
    def get_epoch_augmented_data(self, epoch, ep_start_pitchaug=10, pitchaug_range=(-3, 3)):
        pitchaug_range = [x for x in range(pitchaug_range[0], pitchaug_range[1] + 1)]
        training_data = []
        for seq in self.training_seqs:
            # pitch augmentation
            if epoch >= ep_start_pitchaug:
                seq = deepcopy(seq)
                pitch_change = random.choice( pitchaug_range )
                for i, ev in enumerate(seq):
                    #  event_id = 21 -> Note-On_21 : the lowest pitch on piano
                    if 'Note-On' in self.word2event[ev] and ev >= 21:
                        seq[i] += pitch_change
                    if 'Chord-Tone' in self.word2event[ev]:
                        seq[i] += pitch_change
                        # prevent pitch shift out of range
                        if seq[i] > self.event2word['Chord-Tone_B']:
                            seq[i] -= 12
                        elif seq[i] < self.event2word['Chord-Tone_C']:
                            seq[i] += 12
                    if 'Chord-Slash' in self.word2event[ev]:
                        seq[i] += pitch_change
                        # prevent pitch shift out of range
                        if seq[i] > self.event2word['Chord-Slash_B']:
                            seq[i] -= 12
                        elif seq[i] < self.event2word['Chord-Slash_C']:
                            seq[i] += 12

            # padding sequence to fit the entry length
            if len(seq) < self.entry_len + 1:
                padlen = self.entry_len - len(seq)
                seq.append(1)
                seq.extend([0 for x in range(padlen)])


            # first 10 epoch let the input include start or end of the song
            # -1 for assertion : len(seq) % self.entry_len == 1 (for x,y pair purpose)
            if epoch < 10:
              offset = random.choice([0, (len(seq) % self.entry_len) - 1]) # only 2 possible return value 
            else:
              offset = random.randint(0, (len(seq) % self.entry_len) - 1)  # all entries in the list are possible return value

            assert offset + 1 + self.entry_len * (len(seq) // self.entry_len) <= len(seq)

            seq = seq[ offset : offset + 1 + self.entry_len * (len(seq) // self.entry_len) ]

            assert len(seq) % self.entry_len == 1

            pairs = []
            for i in range(0, len(seq) - self.x_len, self.x_len):
                x, y = seq[i:i+self.x_len], seq[ i+1 : i+self.x_len+1 ]
                assert len(x) == self.x_len
                assert len(y) == self.x_len
                pairs.append([x, y])

            pairs = np.array(pairs)

            # put pairs into training data by groups
            for i in range(0, len(pairs) - self.group_size + 1, self.group_size):
                segment = pairs[i:i+self.group_size]
                assert len(segment) == self.group_size
                training_data.append(segment)

        training_data = np.array(training_data)

        # shuffle training data
        reorder_index = np.arange(len(training_data))
        np.random.shuffle( reorder_index )
        training_data = training_data[ reorder_index ]

        num_batches = len(training_data) // self.batch_size
        # training_data shape (666, 3, 2, 512)
        # training_data shape (group count, self.group_size, pair(x,y), 512)
        print ("training_data.shape , num_batches = {} , {}".format(training_data.shape,num_batches))
        return training_data, num_batches

    ########################################
    # train w/ augmentation
    ########################################
    def train_augment(self, output_checkpoint_folder, pitchaug_range=(-3, 3), logfile=None):

        assert self.training_seqs is not None

        # check output folder
        if not os.path.exists(output_checkpoint_folder):
            os.mkdir(output_checkpoint_folder)

        # check log file folder
        if logfile:
            if not os.path.dirname(logfile) == "":
                os.makedirs(os.path.dirname(logfile),exist_ok=True) 
        
        st = time.time()

        for e in range(1000):
            # one epoch
            # get all data with augmentation
            training_data, num_batches = self.get_epoch_augmented_data(e)
            
            total_loss = []
            for i in range(num_batches):
                # in one batch
                # get one batch data
                segments = training_data[self.batch_size*i:self.batch_size*(i+1)]

                # memory cache for all layers of tranformer
                batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
                
                for j in range(self.group_size):
                    batch_x = segments[:, j, 0, :]
                    batch_y = segments[:, j, 1, :]
                    # prepare feed dict
                    # self.x = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
                    # self.y = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
                    feed_dict = {self.x: batch_x, self.y: batch_y}

                    # self.mems_i a placeholder for memory of all layers in transformer
                    # self.mems_i = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, self.batch_size, self.d_model]) for _ in range(self.n_layer)]
                    
                    

                    for m, m_np in zip(self.mems_i, batch_m):
                        feed_dict[m] = m_np

                    # run
                    _, gs_, loss_, new_mem_ = self.sess.run([self.train_op, self.global_step, self.avg_loss, self.new_mem], feed_dict=feed_dict)
                    
                    batch_m = new_mem_
                    total_loss.append(loss_)
                    # print ('Current lr: {}'.format(self.sess.run(self.optimizer._lr)))
                    print('>>> Epoch: {}, Step: {}, Loss: {:.5f}, Time: {:.2f}'.format(e, gs_, loss_, time.time()-st)) 
            
            print ('[epoch {} avg loss] {:.5f}'.format(e, np.mean(total_loss)))
            if e >= 0:
                self.saver.save(self.sess, '{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))
                if logfile:
                    with open(logfile, 'a') as f:
                        f.write('epoch = {:03d} | loss = {:.5f} | time = {:.2f}\n'.format(e, np.mean(total_loss), time.time()-st))
            # stop
            if np.mean(total_loss) <= 0.05:
                break

    ########################################
    # train
    ########################################
    def train(self, training_data, output_checkpoint_folder):
        # check output folder
        if not os.path.exists(output_checkpoint_folder):
            os.mkdir(output_checkpoint_folder)
        # shuffle
        index = np.arange(len(training_data))
        np.random.shuffle(index)
        training_data = training_data[index]
        num_batches = len(training_data) // self.batch_size
        st = time.time()
        for e in range(1000):
            total_loss = []
            for i in range(num_batches):
                segments = training_data[self.batch_size*i:self.batch_size*(i+1)]
                batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
                for j in range(self.group_size):
                    batch_x = segments[:, j, 0, :]
                    batch_y = segments[:, j, 1, :]
                    # prepare feed dict
                    feed_dict = {self.x: batch_x, self.y: batch_y}
                    for m, m_np in zip(self.mems_i, batch_m):
                        feed_dict[m] = m_np
                    # run
                    _, gs_, loss_, new_mem_ = self.sess.run([self.train_op, self.global_step, self.avg_loss, self.new_mem], feed_dict=feed_dict)
                    batch_m = new_mem_
                    total_loss.append(loss_)
                    # print ('Current lr: {}'.format(self.sess.run(self.optimizer._lr)))
                    print('>>> Epoch: {}, Step: {}, Loss: {:.5f}, Time: {:.2f}'.format(e, gs_, loss_, time.time()-st))

            print ('[epoch {} avg loss] {:.5f}'.format(e, np.mean(total_loss)))
            if not e % 6:
                self.saver.save(self.sess, '{}/model-{:03d}-{:.3f}'.format(output_checkpoint_folder, e, np.mean(total_loss)))
            # stop
            if np.mean(total_loss) <= 0.05:
                break

    ########################################
    # search strategy: temperature (re-shape)
    ########################################
    def temperature(self, logits, temperature):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        return probs

    ########################################
    # search strategy: topk (truncate)
    ########################################
    def topk(self, probs, k):
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:k]
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # search strategy: nucleus (truncate)
    ########################################
    def nucleus(self, probs, p):
        probs /= sum(probs)
        sorted_probs = np.sort(probs)[::-1]
        sorted_index = np.argsort(probs)[::-1]
        cusum_sorted_probs = np.cumsum(sorted_probs)
        after_threshold = cusum_sorted_probs > p
        if sum(after_threshold) > 0:
            last_index = np.where(after_threshold)[0][-1]
            candi_index = sorted_index[:last_index]
        else:
            candi_index = sorted_index[:3] # just assign a value
        candi_probs = [probs[i] for i in candi_index]
        candi_probs /= sum(candi_probs)
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return word

    ########################################
    # inference (for batch size = 1)
    ########################################
    def inference(self, n_bars, strategies, params, use_structure=False, init_mem=None):
        print("Start model inference...")
        # initial start
        words = [[]]
        # add new part if needed
        if use_structure:
            words[-1].append( self.event2word['Part-Start_I'] if random.random() > 0.5 else self.event2word['Part-Start_A'] )
            words[-1].append( self.event2word['Rep-Start_1'])
        # add new bar
        words[-1].append( self.event2word['Bar'] )
        # add position 0
        words[-1].append( self.event2word['Position_0/64'] ) 
        # add random tempo class and bin
        chosen_tempo_cls = random.choice([x for x in range(0, 5)])
        words[-1].append( self.event2word['Tempo-Class_{}'.format(chosen_tempo_cls)] )
        tempo_bin_start = chosen_tempo_cls * 12
        words[-1].append( random.choice(
            [tb for tb in range(self.event2word['Tempo_50.00'] + tempo_bin_start, self.event2word['Tempo_50.00'] + tempo_bin_start + 12)]
        ))
        # add random chord
        if not use_structure and random.random() > 0.5 or \
           use_structure and words[-1][0] == self.event2word['Part-Start_A']:
            words[-1].append( random.choice(
                [ct for ct in range(self.event2word['Chord-Tone_C'], self.event2word['Chord-Tone_C'] + 12)]
            ))
            words[-1].append( random.choice(
                [self.event2word[ct] for ct in self.event2word.keys() if 'Chord-Type' in ct]
            ))
        # initialize mem
        if init_mem is None:
            batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
        else:
            batch_m = init_mem
        
        # generate

        
        initial_flag, initial_cnt = True, 0
        generated_bars = 0

        # define legal beat posisition
        beat_pos = set(['Position_0/64', 'Position_16/64', 'Position_32/64', 'Position_48/64'])

        allowed_pos = set([x for x in range(self.event2word['Position_0/64'] + 1, self.event2word['Position_0/64'] + 17)])
        fail_cnt = 0
        
        while generated_bars < n_bars:
            print("Generating bars #{}/{}".format(generated_bars+1,n_bars), end='\r')
            if fail_cnt:
                print ('failed iterations:', fail_cnt)
            
            if fail_cnt > 256:
                print ('model stuck ...')
                exit()

            # prepare input
            if initial_flag:
                temp_x = np.zeros((self.batch_size, len(words[0])))
                for b in range(self.batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = False
            else:
                temp_x = np.zeros((self.batch_size, 1))
                for b in range(self.batch_size):
                    temp_x[b][0] = words[b][-1]

            # prepare feed dict
            # inside a feed dict
            # placeholder : data
            # put input into feed_dict
            feed_dict = {self.x: temp_x}

            # put memeory into feed_dict
            for m, m_np in zip(self.mems_i, batch_m):
                feed_dict[m] = m_np
            
            # model (prediction)
            _logits, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=feed_dict)

            logits = _logits[-1, 0]

            # temperature or not
            if 'temperature' in strategies:
                if initial_flag:
                    probs = self.temperature(logits=logits, temperature=1.5)
                else:
                    probs = self.temperature(logits=logits, temperature=params['t'])
            else:
                probs = self.temperature(logits=logits, temperature=1.)

            # sampling
            # word : the generated remi event
            word = self.nucleus(probs=probs, p=params['p'])
            # print("Generated new remi word {}".format(word))
            # skip padding
            if word in [0, 1]:
                fail_cnt += 1
                continue
            
            # illegal sequences 
            # words[0][-1] : last generated word
            if 'Bar' in self.word2event[words[0][-1]] and self.word2event[word] != 'Position_0/64':
                fail_cnt += 1
                continue
            if self.word2event[words[0][-1]] in beat_pos and 'Tempo-Class' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Tempo-Class' in self.word2event[words[0][-1]] and 'Tempo_' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Note-Velocity' in self.word2event[words[0][-1]] and 'Note-On' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Note-On' in self.word2event[words[0][-1]] and 'Note-Duration' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Chord-Tone' in self.word2event[words[0][-1]] and 'Chord-Type' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Chord-Type' in self.word2event[words[0][-1]] and 'Chord-Slash' not in self.word2event[word]:
                fail_cnt += 1
                continue
            if 'Position' in self.word2event[word] and word not in allowed_pos:
                fail_cnt += 1
                continue
            if self.word2event[word].split('_')[0] == self.word2event[words[0][-1]].split('_')[0]:
                fail_cnt += 1
                continue

            
            # update allowed positions
            # if the new word is a beat event then we need to update the new allowed_pos
            # ex if the new word is (209: Position_16/64) then the allow_pos should update as following
            # ['Position_1/64' to  'Position_16/64']  -> ['Position_17/64' to  'Position_32/64']
            # exception: exceed the 64/64 go back to 0
            if self.word2event[word] in beat_pos:
                if self.word2event[word] == 'Position_48/64':
                    allowed_pos = set([x for x in range(self.event2word['Position_49/64'], self.event2word['Position_49/64'] + 15)] + [self.event2word['Position_0/64']])
                else:
                    allowed_pos = set([x for x in range(word + 1, word + 17)])

            # add new event to record sequence
            words[0].append(word)
            fail_cnt = 0

            # record n_bars
            if word == self.event2word['Bar']:
                generated_bars += 1
            # re-new mem
            batch_m = _new_mem

        print ('generated {} events'.format(len(words[0])))
        return words[0]

    ########################################
    # close
    ########################################
    def close(self):
        self.sess.close()
