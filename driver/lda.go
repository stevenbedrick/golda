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
	
	
	// handle option parsing
	// hard-coded for now
	sample_fname := "/Users/steven/Dropbox/Steven/LDA/mi.gerd.bc.2011.title.abstract.txt"
	// sample_fname := "/Users/steven/Dropbox/Steven/LDA/mi.gerd.bc.2011.txt"	
	// sample_fname := "/Users/steven/Dropbox/Steven/LDA/test_input.txt"	
	
	// load input file into list of strings
	contents, err := ioutil.ReadFile(sample_fname)
	if err != nil {
		fmt.Println("Problem reading file: " + err.Error())
		return
	}
	
	lines := strings.Split(string(contents), "\n")		
	
	// processing- lowercase, downsampling, normalizing, pruning, stopword removal, etc.
	should_lowercase := true // TODO: get from optparse
	remove_stopwords := true
	remove_non_words := true
	remove_solo_digits := true
	
	// turn list of strings into token vectors
	var tokenss [][]string
	for _, s := range(lines) {
		if len(strings.TrimSpace(s)) > 0 {			
			temp_tokens := util.Tokenize(s, should_lowercase)
			if remove_stopwords {
				temp_tokens = util.FilterStopwords(temp_tokens)
			}
			
			if remove_non_words {
				temp_tokens = util.FilterNonAlpha(temp_tokens)
			}
			
			if remove_solo_digits {
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
	ntopics := 10
	alpha := 0.01
	beta := 0.01
	iterations := 1000
	repetitions := 5
	seed := 0
	parallel := true
	report_token_limit := 10
	
	l := model.NewModeler(ntopics, alpha, beta, iterations, repetitions, seed, parallel)
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
		for i := len(t_probs) - 1; i >= len(t_probs) - report_token_limit; i-- {
			this_tok_prob := t_probs[i]
			fmt.Printf("\t%s = %0.4f\n", this_tok_prob.Token, this_tok_prob.Probability)
		} 
	}


}