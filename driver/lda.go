package main

import (
	"fmt"
	"io/ioutil"
	"strings"
	"lda/model"
	"lda/util"
	"time"
	"sort"
	"runtime"
	"flag"
	// "runtime/pprof"
	// "log"
	// "os"
)



func main() {
	
	// pprof
    // f, err := os.Create("prof.out")
    // if err != nil {
    //     log.Fatal(err)
    // }
    // pprof.StartCPUProfile(f)
    // defer pprof.StopCPUProfile()
	
	// load input file into list of strings

	var default_file = "/Users/steven/Dropbox/Steven/LDA/mi.gerd.bc.2011.title.abstract.txt"
	// var default_file = "/Users/steven/Dropbox/Steven/LDA/mi.gerd.bc.2011.txt"
	// var default_file = "/Users/steven/Dropbox/Steven/LDA/test_input.txt"	

	var sample_fname = flag.String("input", default_file, "Input file to use. One line per record.")
	var should_lowercase = flag.Bool("lowercase", true, "Lowercase text as pre-processing step.")
	var remove_stopwords = flag.Bool("stopwords", true, "Remove common English stopwords as a pre-processing step.")
	var remove_non_words = flag.Bool("remove_non_words", true, "Remove tokens that contain non-alphanumeric ([\\w]) characters. NOT recommended in Unicode-sensitive contexts.")
	var remove_solo_digits = flag.Bool("remove_solo_digits", true, "Remove tokens that consist entirely of digits.")
	var ntopics = flag.Int("ntopics", 10, "Number of topics.")
	var alpha = flag.Float64("alpha", 0.01, "Alpha parameter- hyperparameter for Dirichelt prior on the document-topic distribution.")
	var beta = flag.Float64("beta", 0.01, "Beta parameter- hyperparameter for Dirichlet prior on topic-word distribution.")
	var iterations = flag.Int("iterations", 100, "Number of iterations to run.")
	var repetitions = flag.Int("repetitions", 5, "Number of repetitions to average.")
	var seed = flag.Int("seed", 0, "RNG seed. Probably best left alone.")
	var parallel = flag.Bool("parallel", true, "Process repetitions in parallel.")
	var report_token_limit = flag.Int("report_token_limit", 10, "Number of tokens to report.")

	flag.Parse()

	contents, err := ioutil.ReadFile(*sample_fname)
	if err != nil {
		fmt.Println("Problem reading file: " + err.Error())
		return
	}
	
	lines := strings.Split(string(contents), "\n")		
	
	// processing- lowercase, downsampling, normalizing, pruning, stopword removal, etc.
	// turn list of strings into token vectors
	var tokenss [][]string
	for _, s := range(lines) {
		if len(strings.TrimSpace(s)) > 0 {			
			temp_tokens := util.Tokenize(s, *should_lowercase)
			if *remove_stopwords {
				temp_tokens = util.FilterStopwords(temp_tokens)
			}
			
			if *remove_non_words {
				temp_tokens = util.FilterNonAlpha(temp_tokens)
			}
			
			if *remove_solo_digits {
				temp_tokens = util.FilterOnlyNumbers(temp_tokens)
			}
			
			tokenss = append(tokenss, temp_tokens)
		}
	}
	
	fmt.Println("num samples: ", len(tokenss))

	// s t gomaxprocs
	runtime.GOMAXPROCS(7) //runtime.NumCPU())
	
	// get rid of common tokens if needed

	// 1. Are we building a new model or applying existing?
	
	// New model:
	// TODO: all should be configured using option parsing
	// ntopics := 10
	// alpha := 0.01
	// beta := 0.01
	// iterations := 1000
	// repetitions := 5
	// seed := 0
	// parallel := true
	// report_token_limit := 10
	
	l := model.NewModeler(*ntopics, *alpha, *beta, *iterations, *repetitions, *seed, *parallel)
	start_time := time.Now()
	l.Model(tokenss)
	fmt.Printf("Elapsed time: %s\n", time.Since(start_time))
	// Existing model:
	
	// Reporting results
	for topic, dx := range(l.GetTopicTokenProbabilityAssignments()) {
		fmt.Printf("TOPIC %d:\n", topic)
		t_probs := model.TokenProbsFromMap(dx)
		sort.Sort(model.ByProb{t_probs})
		// Sort() goes in ascending order, and we want descending- so walk backwards from the end
		for i := len(t_probs) - 1; i >= len(t_probs) - *report_token_limit; i-- {
			this_tok_prob := t_probs[i]
			fmt.Printf("\t%s = %0.4f\n", this_tok_prob.Token, this_tok_prob.Probability)
		} 
	}


}