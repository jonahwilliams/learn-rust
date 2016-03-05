#![allow(dead_code)]
extern crate csv;
extern crate rand;
//#[macro_use] extern crate itertools;

use std::collections::HashMap;
use rand::distributions::{IndependentSample, Range};
//use itertools::Itertools;

type Encoder = HashMap<String, u32>;
type Decoder = HashMap<u32, String>;

/*
    LDA Model
    K: Number of topics
    w: Vector of document word vectors
    t: Vector of document topic vectors
*/
struct LDA {
    k: u32,
    words: Vec<Vec<u32>>,
    topics: Vec<Vec<u32>>,
    encoder: Encoder,
    decoder: Decoder
}

impl LDA {
    fn new(num_topics: u32, vocab_file: &str, dataset: Vec<String>) -> LDA {
        // Encoder/Decoder HashMap pair
        let (enc, dec): (Encoder, Decoder) = LDA::load_vocab(vocab_file);

        // create and cache rng and between for generating initial topics
        let between = Range::new(0, num_topics);
        let mut rng = rand::thread_rng();

        for item in &dataset {
            print!("{}", item);
        }

        // dataset is provided externally, and here is represented by a vector
        // of strings.  map over each one and tokenize it with the encoder.
        let w: Vec<Vec<u32>> = dataset
            .iter()
            .map(|x| LDA::tokenize(x, &enc))
            .collect::<Vec<Vec<u32>>>();

        // Randomly initialize a matching vector of topics
        let mut t: Vec<Vec<u32>> = Vec::with_capacity(w.len());
        for i in 0..w.len() {
            let mut tt = Vec::with_capacity(w[i].len());
            for j in 0..w[i].len() {
                tt[j] = between.ind_sample(&mut rng);
            }
            t[i] = tt;
        }

        LDA {
            k: num_topics,
            words: w,
            topics: t,
            encoder: enc,
            decoder: dec
        }
    }

    /*
        At each step we calculate p(topic t | document d) = the proportion of words in
        document d that are currently assigned to topic t.
        p(word w | topic t) = the proportion of assignments to
        topic t over all documents that come from this word w
        NOTE: this implementation is slow as shit.
    */
    fn sample(&mut self) {
        let mut rng = rand::thread_rng();
        let reroll = Range::new(0.0, 1.0);
        let between = Range::new(0, self.k);
        let n = self.words.len();

        for i in 0..n {
            let m = self.words[i].len();
            for j in 0..m {
                let t = self.topics[i][j];
                let w = self.words[i][j];
                let mut topic_count = 0;
                let mut word_topic_count = 0;
                let mut word_count = 0;


                for jj in 0..m {
                    if self.topics[i][jj] == t {
                        topic_count += 1;
                    }
                }

                for ii in 0..n {
                    for jjj in 0..m {
                        if self.words[ii][jjj] == w {
                            word_count += 1;
                            if self.topics[ii][jjj] == t {
                                word_topic_count += 1;
                            }
                        }
                    }
                }

                let ptd: f64 = topic_count as f64 / self.topics[i].len() as f64;
                let pwt: f64 = word_topic_count as f64 / word_count as f64;
                if reroll.ind_sample(&mut rng) > ptd * pwt {
                    self.topics[i][j] = between.ind_sample(&mut rng);
                }
            }
        }
    }

    /* create two HashMaps to encode and decode strings into u32
     * loads external csv file which is just a list of words
     * because LDA is a bag of words model, unknown strings are
     * discarded.
     */
    fn load_vocab(file_name: &str) -> (Encoder, Decoder) {
        let mut rdr = csv::Reader::from_file(file_name).unwrap();
        let mut index: u32 = 1;
        let mut encoder: Encoder = HashMap::new();
        let mut decoder: Decoder = HashMap::new();

        for row in rdr.decode() {
            let word: String = row.unwrap();
            encoder.insert(word.to_string(), index);
            decoder.insert(index, word.to_string());
            index += 1;
        }
        (encoder, decoder)
    }

    // Takes a reference to a string and a reference to the encoder
    // HashMap and produces a vector of word tokens.  unmmatched
    // strings are ignored
    fn tokenize(phrase: &str, encoder: &Encoder) -> Vec<u32> {
        phrase
            .split("\\P{L}+")
            .map(|x| match encoder.get(x) {
                Some(y) => y.clone(),
                None => 0
            })
            .filter(|&x| x > 0)
            .collect()
    }
}

fn main() {
    let data = vec!(
        "I've heard this advice over and over and over at startup events".to_string(),
        "to the point that I got a little sick of hearing it. It's not wrong.".to_string(),
        "to the point that I got a little sick of hearing it. It's not wrong.".to_string()
    );
    // data[1] = "to the point that I got a little sick of hearing it. It's not wrong.".to_string();
    // data[2] = "Putting aside the fact that every single other startup in the world who".to_string();
    // data[3] = "heard this same advice before you is already out there frantically doing".to_string();
    // data[4] = "everything they can to hire all the best people out from under you and everyone else".to_string();
    // data[5] = "it is superficially true. A company staffed by a bunch of people who don't care".to_string();
    // data[6] = "about their work and aren't good at their jobs isn't exactly poised for success.".to_string();
    // data[7] = "But in a room full of people giving advice to startups, nobody wants to talk about".to_string();
    // data[8] = "the elephant in that room".to_string();
    let lda = LDA::new(32, "./data/data.csv", data);
    return;
}
