# Emojify!

Project uses word vector representations to build an Emojifier.

So rather than writing "Congratulations on the promotion! Lets get coffee and talk. Love you!" the emojifier automatically turn them into "Congratulations on the promotion! 👍 Lets get coffee and talk. ☕️ Love you! ❤️"

model takes inputs a sentence (such as "Let's go see the baseball game tonight!") and finds the most appropriate emoji to be used with this sentence (⚾️). In many emoji interfaces, remember that ❤️ is the "heart" symbol rather than the "love" symbol.

It can be see that even if the training set explicitly relates only a few words to a particular emoji, the algorithm will generalize and associate words in the test set to the same emoji even if those words don't even appear in the training set. This allows us to build an accurate classifier mapping from sentences to emojis, even using a small training set.


## 1 - Baseline model: Emojifier-V1

### 1.1 - Dataset EMOJISET

We have a tiny dataset (X, Y) where:
- X contains 127 sentences (strings)
- Y contains a integer label between 0 and 4 corresponding to an emoji for each sentence

<img src="https://raw.githubusercontent.com/00arun00/Emojify/master/images/data_set.png" style="width:700px;height:300px;">
<caption><center> **Figure 1**: EMOJISET - a classification problem with 5 classes. A few examples of sentences are given here. </center></caption>

split the dataset between training (127 examples) and testing (56 examples).

### 1.2 - Overview of the Emojifier-V1

Figure below shows the baseline model called "Emojifier-v1".  

<center>
<img src="https://raw.githubusercontent.com/00arun00/Emojify/master/images/image_1.png" style="width:900px;height:300px;">
<caption><center> **Figure 2**: Baseline model (Emojifier-V1).</center></caption>
</center>

The input of the model is a string corresponding to a sentence (e.g. "I love you). In the code, the output will be a probability vector of shape (1,5), that can then be passed to an argmax layer to extract the index of the most likely emoji output.

### 1.3 - Implementing Emojifier-V1

As shown in Figure (2), the first step is to convert an input sentence into the word vector representation, which then get averaged together. Here we will use pretrained 50-dimensional GloVe embeddings. Run the following cell to load the `word_to_vec_map`, which contains all the vector representations.

#### Model

![eq2](http://latex.codecogs.com/gif.latex?%24%24%20z%5E%7B%28i%29%7D%20%3D%20W%20.%20avg%5E%7B%28i%29%7D%20&plus;%20b%24%24)    
![eq3](http://latex.codecogs.com/gif.latex?%24%24%20a%5E%7B%28i%29%7D%20%3D%20softmax%28z%5E%7B%28i%29%7D%29%24%24)    
![eq2](http://latex.codecogs.com/gif.latex?%24%24%20%5Cmathcal%7BL%7D%5E%7B%28i%29%7D%20%3D%20-%20%5Csum_%7Bk%20%3D%200%7D%5E%7Bn_y%20-%201%7D%20Yoh%5E%7B%28i%29%7D_k%20*%20log%28a%5E%7B%28i%29%7D_k%29%24%24)    

### 1.4 Results
```
Training set:
Accuracy: 0.9772727272727273
Test set:
Accuracy: 0.8571428571428571
```
Random guessing would have had 20% accuracy given that there are 5 classes. This is pretty good performance after training on only 127 examples.

#### Confusion Matrix
<img src="https://raw.githubusercontent.com/00arun00/Emojify/master/images/confusion.png" style="width:250px;height:250px;"> <br>
<caption><center> **Figure 3**: Confusion Matrix for Emojify-V1. </center></caption>


## 2 - Emojifier-V2: Using LSTMs:

Here we build an LSTM model that takes as input word sequences. This model will be able to take word ordering into account. Emojifier-V2 will continue to use pre-trained word embeddings to represent words, but will feed them into an LSTM, whose job it is to predict the most appropriate emoji.

### 2.1 - model

Emojifier-v2:

<img src="https://raw.githubusercontent.com/00arun00/Emojify/master/images/emojifier-v2.png" style="width:700px;height:400px;"> <br>
<caption><center> **Figure 4**: Emojifier-V2. A 2-layer LSTM sequence classifier. </center></caption>

### 2.2 Keras and mini-batching

If you have a 3-word sentence and a 4-word sentence, then the computations needed for them are different (one takes 3 steps of an LSTM, one takes 4 steps) so it's just not possible to do them both at the same time.

The common solution to this is to use padding. Specifically, set a maximum sequence length, and pad all sequences to the same length. For example, of the maximum sequence length is 20, we could pad every sentence with "0"s so that each input sentence is of length 20. Thus, a sentence "i love you" would be represented as ![eq1](http://latex.codecogs.com/gif.latex?%24%28e_%7Bi%7D%2C%20e_%7Blove%7D%2C%20e_%7Byou%7D%2C%20%5Cvec%7B0%7D%2C%20%5Cvec%7B0%7D%2C%20%5Cldots%2C%20%5Cvec%7B0%7D%29%24). In this example, any sentences longer than 20 words would have to be truncated. One simple way to choose the maximum sequence length is to just pick the length of the longest sentence in the training set.

### 2.3 - The Embedding layer

In Keras, the embedding matrix is represented as a "layer", and maps positive integers (indices corresponding to words) into dense vectors of fixed size (the embedding vectors). It can be trained or initialized with a pretrained embedding. Here we create an [Embedding()](https://keras.io/layers/embeddings/) layer in Keras, initialize it with the GloVe 50-dimensional vectors loaded earlier in the notebook. Because our training set is quite small, we will not update the word embeddings but will instead leave their values fixed. But in the code below, we'll show you how Keras allows you to either train or leave fixed this layer.  

The `Embedding()` layer takes an integer matrix of size (batch size, max input length) as input. This corresponds to sentences converted into lists of indices (integers), as shown in the figure below.

<img src="https://raw.githubusercontent.com/00arun00/Emojify/master/images/embedding1.png" style="width:700px;height:250px;">
<caption><center> **Figure 5**: Embedding layer. This example shows the propagation of two examples through the embedding layer. Both have been zero-padded to a length of `max_len=5`. The final dimension of the representation is  `(2,max_len,50)` because the word embeddings we are using are 50 dimensional. </center></caption>

The largest integer (i.e. word index) in the input should be no larger than the vocabulary size. The layer outputs an array of shape (batch size, max input length, dimension of word vectors).

The first step is to convert all your training sentences into lists of indices, and then zero-pad all these lists so that their length is the length of the longest sentence.

## 2.3 Building the Emojifier-V2

Lets now build the Emojifier-V2 model.

<img src="https://raw.githubusercontent.com/00arun00/Emojify/master/images/emojifier-v2.png" style="width:700px;height:400px;"> <br>
<caption><center> **Figure 6**: Emojifier-v2. A 2-layer LSTM sequence classifier. </center></caption>

model is compiled using `categorical_crossentropy` loss, `adam` optimizer and `['accuracy']` metrics:

## 2.4 Results
```
Train accuracy = 0.9848
Test accuracy =  0.9286

Examples that were labeled correctly:

Expected emoji:🍴 prediction: I want to eat	🍴
Expected emoji:😞 prediction: he did not answer	😞
Expected emoji:😄 prediction: he got a very nice raise	😄
Expected emoji:😄 prediction: she got me a nice present	😄
Expected emoji:😄 prediction: ha ha ha it was so funny	😄
Expected emoji:😄 prediction: he is a good friend	😄
Expected emoji:😞 prediction: I am upset	😞
Expected emoji:😄 prediction: We had such a lovely dinner tonight	😄
Expected emoji:🍴 prediction: where is the food	🍴
Expected emoji:😄 prediction: Stop making this joke ha ha ha	😄
Expected emoji:⚾ prediction: where is the ball	⚾
Expected emoji:😞 prediction: are you serious😞
Expected emoji:⚾ prediction: Let us go play baseball	⚾
Expected emoji:😞 prediction: This stupid grader is not working 	😞
Expected emoji:😞 prediction: work is horrible	😞
Expected emoji:😄 prediction: Congratulation for having a baby	😄
Expected emoji:😞 prediction: stop pissing me off😞
Expected emoji:🍴 prediction: any suggestions for dinner	🍴
Expected emoji:🍴 prediction: I boiled rice	🍴
Expected emoji:😞 prediction: she is a bully	😞
Expected emoji:😞 prediction: Why are you feeling bad	😞
Expected emoji:😞 prediction: I am upset	😞
Expected emoji:⚾ prediction: give me the ball⚾
Expected emoji:❤️ prediction: My grandmother is the love of my life	❤️
Expected emoji:⚾ prediction: enjoy your game⚾
Expected emoji:😄 prediction: valentine day is near	😄
Expected emoji:❤️ prediction: I miss you so much	❤️
Expected emoji:⚾ prediction: throw the ball	⚾
Expected emoji:😞 prediction: My life is so boring	😞
Expected emoji:😄 prediction: she said yes	😄
Expected emoji:😄 prediction: will you be my valentine	😄
Expected emoji:⚾ prediction: he can pitch really well	⚾
Expected emoji:😄 prediction: dance with me	😄
Expected emoji:🍴 prediction: I am hungry🍴
Expected emoji:🍴 prediction: See you at the restaurant	🍴
Expected emoji:😄 prediction: I like to laugh	😄
Expected emoji:⚾ prediction: I will  run⚾
Expected emoji:❤️ prediction: I like your jacket 	❤️
Expected emoji:❤️ prediction: i miss her	❤️
Expected emoji:⚾ prediction: what is your favorite baseball game	⚾
Expected emoji:😄 prediction: Good job	😄
Expected emoji:❤️ prediction: I love you to the stars and back	❤️
Expected emoji:😄 prediction: What you did was awesome	😄
Expected emoji:😄 prediction: ha ha ha lol	😄
Expected emoji:😞 prediction: I do not want to joke	😞
Expected emoji:😞 prediction: go away	😞
Expected emoji:😞 prediction: yesterday we lost again	😞
Expected emoji:❤️ prediction: family is all I have	❤️
Expected emoji:😞 prediction: you are failing this exercise	😞
Expected emoji:😄 prediction: Good joke	😄
Expected emoji:😄 prediction: You deserve this nice prize	😄
Expected emoji:🍴 prediction: I did not have breakfast 🍴

Examples which have been miss predicted:

Expected emoji:😞 prediction: work is hard	😄
Expected emoji:😞 prediction: This girl is messing with me	❤️
Expected emoji:❤️ prediction: I love taking breaks	😞
Expected emoji:😄 prediction: you brighten my day	❤️
```
