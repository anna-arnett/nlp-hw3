# NLP HW3

Please find the homework assignment instructions [here](https://docs.google.com/document/d/1K8s_Ecms0cIqRO1PKPFs2bfFVFfZpc1nFoEhtxRlCaM/edit?tab=t.eia9bivtc3n8).

## Part 1
* English tokenizer’s output for the first 10 sentences of test.en.txt:

=== First 10 tokenized English test sentences (with BOS/EOS) ===
00 INDICES: [0, 29, 92, 82, 83, 384, 256, 4598, 96, 465, 14, 1]
00 TOKENS: ['<BOS>', 'A', 'man', 'in', 'an', 'orange', 'hat', 'starring', 'at', 'something', '.', '<EOS>']
01 INDICES: [0, 29, 6127, 9188, 100, 352, 88, 4637, 263, 343, 82, 229, 103, 56, 167, 806, 14, 1]
01 TOKENS: ['<BOS>', 'A', 'Boston', 'Terrier', 'is', 'running', 'on', 'lush', 'green', 'grass', 'in', 'front', 'of', 'a', 'white', 'fence', '.', '<EOS>']
02 INDICES: [0, 29, 174, 82, 1952, 518, 6108, 56, 695, 108, 56, 229, 845, 14, 1]
02 TOKENS: ['<BOS>', 'A', 'girl', 'in', 'karate', 'uniform', 'breaking', 'a', 'stick', 'with', 'a', 'front', 'kick', '.', '<EOS>']
03 INDICES: [0, 906, 158, 155, 1189, 1395, 95, 1526, 156, 82, 93, 297, 12, 108, 297, 2493, 496, 82, 93, 414, 14, 1]
03 TOKENS: ['<BOS>', 'Five', 'people', 'wearing', 'winter', 'jackets', 'and', 'helmets', 'stand', 'in', 'the', 'snow', ',', 'with', 'snow', 'mobi', 'les', 'in', 'the', 'background', '.', '<EOS>']
04 INDICES: [0, 354, 119, 1683, 93, 1274, 103, 56, 902, 14, 1]
04 TOKENS: ['<BOS>', 'People', 'are', 'fixing', 'the', 'roof', 'of', 'a', 'house', '.', '<EOS>']
05 INDICES: [0, 29, 92, 82, 638, 885, 692, 2591, 56, 213, 103, 133, 155, 598, 1196, 95, 829, 204, 363, 56, 126, 338, 82, 56, 2860, 584, 148, 3145, 14, 1]
05 TOKENS: ['<BOS>', 'A', 'man', 'in', 'light', 'colored', 'clothing', 'photographs', 'a', 'group', 'of', 'men', 'wearing', 'dark', 'suits', 'and', 'hats', 'standing', 'around', 'a', 'woman', 'dressed', 'in', 'a', 'stra', 'ple', 'ss', 'gown', '.', '<EOS>']
06 INDICES: [0, 29, 213, 103, 158, 204, 82, 229, 103, 83, 8881, 14, 1]
06 TOKENS: ['<BOS>', 'A', 'group', 'of', 'people', 'standing', 'in', 'front', 'of', 'an', 'igloo', '.', '<EOS>']
07 INDICES: [0, 29, 175, 82, 56, 153, 518, 100, 1627, 109, 6146, 782, 218, 96, 1155, 1726, 12, 183, 93, 3091, 82, 93, 190, 518, 100, 1627, 109, 710, 498, 14, 1]
07 TOKENS: ['<BOS>', 'A', 'boy', 'in', 'a', 'red', 'uniform', 'is', 'attempting', 'to', 'avoid', 'getting', 'out', 'at', 'home', 'plate', ',', 'while', 'the', 'catcher', 'in', 'the', 'blue', 'uniform', 'is', 'attempting', 'to', 'catch', 'him', '.', '<EOS>']
08 INDICES: [0, 29, 515, 1177, 88, 56, 330, 14, 1]
08 TOKENS: ['<BOS>', 'A', 'guy', 'works', 'on', 'a', 'building', '.', '<EOS>']
09 INDICES: [0, 29, 92, 82, 56, 879, 100, 196, 82, 56, 540, 95, 235, 3667, 14, 1]
09 TOKENS: ['<BOS>', 'A', 'man', 'in', 'a', 'vest', 'is', 'sitting', 'in', 'a', 'chair', 'and', 'holding', 'magazines', '.', '<EOS>']

* Free response: I noticed that in general, it does a pretty good job of splitting up the words into tokens. The words that ended up being broken into separate tokens despite being one word (snowmobiles, strapless) are longer words that are either plural or could easily be confused as plural (i.e., strapless). But overall, I thought the splitting was pretty good. 

## Part 3
* BLEU: 24.78
* ChrF: 44.98
* TER: 65.72
* Free response:

a) Looking at the first 20 sentences of the test set, the model is generally outputting short but coherent sentences with a typical sentence structure. When it sees patterns that show up frequently, it handles them pretty well and does a good job on it. It struggles a little bit with words that are less common or sentences that are longer, but I think that is expected for something like this. 

b) I think all three are definitely helpful, and BLEU and TER are good for rough comparisons, but I think ChrF is the best one for these shorter sentences. Since ChrF does things on a character level, it isn't as sensitive to smaller differences in inflection or things like that, which I think is really helpful for this task. 
