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
)


func main() {
	
	// handle option parsing
	// hard-coded for now
	sample_fname := "/Users/steven/Dropbox/Steven/LDA/mi.gerd.bc.2011.txt"	
	
	// load input file into list of strings
	contents, err := ioutil.ReadFile(sample_fname)
	if err != nil {
		fmt.Println("Problem reading file: " + err.Error())
		return
	}
	
	lines := strings.Split(string(contents), "\n")		
	
	// processing- lowercase, downsampling, normalizing, pruning, stopword removal, etc.
	report_token_limit := 5
	
	// turn list of strings into token vectors
	var tokenss [][]string
	for _, s := range(lines) {
		if len(strings.TrimSpace(s)) > 0 {
			tokenss = append(tokenss, util.Tokenize(s))
		}
	}
	
	fmt.Println("num samples: ", len(tokenss))
	
	// set gomaxprocs
	runtime.GOMAXPROCS(2)
	
	// get rid of common tokens if needed

	// 1. Are we building a new model or applying existing?
	
	// New model:
	// TODO: all should be configured using option parsing
	l := model.NewModeler(5, 1.0, 1.0, 1000, 1, 0)
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