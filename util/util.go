package util

import (
	"strings"
	"regexp"
	"fmt"
)
// Utility functions- some ported from Python


func TokenMaker(dat []string, f func(string)string) []Token {
	to_ret := make([]Token, len(dat))
	for idx, s := range(dat) {		
		to_ret[idx] = Token(f(s))
	}
	return to_ret
}

// implements SPUD's simple tokenization algorithm
func Tokenize(s string, lowercase bool) Document {
	patt, _ := regexp.Compile("[\\.\\?\\!,:;\\(\\)\\'\\\"]")
	proc_s := patt.ReplaceAllString(s, " ")
	if lowercase {
		proc_s = strings.ToLower(proc_s)
	}
	return TokenMaker(strings.Fields(proc_s), strings.TrimSpace)
}

// TODO: maybe re-write in a more "generator"-y way, with channels? 

func FilterStopwords(t_list []Token) []Token {
	to_ret := []Token{}
	for _, t := range(t_list) {
		if !IsStopWord(t) {
			to_ret = append(to_ret, t)
		}
	}
	return to_ret
}

// get rid of anything that doesn't match a regex
func FilterRegex(t_list []Token, patt *regexp.Regexp) []Token {
	to_ret := []Token{}
	for _, t := range(t_list) {
		if patt.MatchString(string(t)) {
			to_ret = append(to_ret, t)
		}
	}
	return to_ret
}

func FilterNonAlpha(t_list []Token) []Token {
	word_chars, _ := regexp.Compile("^\\w+$")
	return FilterRegex(t_list, word_chars)
}

func FilterOnlyNumbers(t_list []Token) []Token {
	only_digits, _ := regexp.Compile("^[^\\d]+$")
	return FilterRegex(t_list, only_digits)
}

// TODO: is there a better way to handle this?
func SumFloat(v []float64) float64 {
	to_return := 0.0
	for _, i := range(v) {
		to_return += i
	}
	return to_return
}

func SumInt(v []int) int {
	to_return := 0
	for _, i := range(v) {
		to_return += i
	}
	return to_return
}

func SumVectors(v1 []float64, v2 []float64) ([]float64, error ){
	if len(v1) != len(v2) {
		return nil, fmt.Errorf("Vectors must be same length; %d and %d don't match.", len(v1), len(v2))
	}
	
	to_return := make([]float64, len(v1))
	for i := 0; i < len(v1); i++ {
		to_return[i] = v1[i] + v2[i]
	}
	
	return to_return, nil	
}

func L1NormalizeVector(v []float64) []float64 {
	to_return := make([]float64, len(v))
	s := SumFloat(v)
	if s == 0.0 {
		return v
	} else {
		s = 1.0/s
		for idx, i := range(v) {
			to_return[idx] = i * s
		}
	}
	return to_return
}

// emulates Python's range(int) behavior.
func PyRange(n int) []int {
	to_return := make([]int, n)
	for i := 0; i < n; i++ {
		to_return[i] = i
	}
	return to_return
}

// gets a slice containing the keys in a string/int map
func KeysFromMap(m map[Token]int) []Token {
	to_return := []Token{}
	for k, _ := range(m) {
		to_return = append(to_return, k)
	}
	return to_return
}

// takes a slice of string-slices and returns a set of the unique elements
func SetFromLists(lists [][]Token) []Token {
	seen := map[Token]bool{}
	for _, list := range(lists) {
		for _, str := range(list) {
			seen[str] = true
		}
	}
	to_return := []Token{}
	for k, _ := range(seen) {
		to_return = append(to_return, k)
	}
	return to_return
}

func SumAndNormalizeListOfDicts(dxlist []TokenProbMap) TokenProbMap {
	total := Probability(0.0)
	total_dx := make(TokenProbMap)
	for _, dx := range(dxlist) {
		for t, prob := range(dx) {			
			total_dx[t] += prob
			total += prob
		}
	}
	dx := make(TokenProbMap)
	for t, prob := range(total_dx) {
		dx[t] = prob / total
	}
	return dx
}

