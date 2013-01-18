package util

import (
	"strings"
	"regexp"
	"fmt"
)
// Utility functions- some ported from Python


// maps a string->string function over a slice of strings
func StringMap(dat []string, f func(string)string) []string {
	to_ret := make([]string, len(dat))
	for idx, s := range(dat) {		
		to_ret[idx] = f(s)
	}
	return to_ret
}

// implements SPUD's simple tokenization algorithm
func Tokenize(s string, lowercase bool) []string {
	patt, _ := regexp.Compile("[\\.\\?\\!,:;\\(\\)\\'\\\"]")
	proc_s := patt.ReplaceAllString(s, " ")
	if lowercase {
		proc_s = strings.ToLower(proc_s)
	}
	return StringMap(strings.Fields(proc_s), strings.TrimSpace)
}

// TODO: maybe re-write in a more "generator"-y way, with channels? 

func FilterStopwords(s_list []string) []string {
	to_ret := []string{}
	for _, s := range(s_list) {
		if !IsStopWord(s) {
			to_ret = append(to_ret, s)
		}
	}
	return to_ret
}

// get rid of anything that doesn't match a regex
func FilterRegex(s_list []string, patt *regexp.Regexp) []string {
	to_ret := []string{}
	for _, s := range(s_list) {
		if patt.MatchString(s) {
			to_ret = append(to_ret, s)
		}
	}
	return to_ret
}

func FilterNonAlpha(s_list []string) []string {
	word_chars, _ := regexp.Compile("^\\w+$")
	return FilterRegex(s_list, word_chars)
}

func FilterOnlyNumbers(s_list []string) []string {
	only_digits, _ := regexp.Compile("^[^\\d]+$")
	return FilterRegex(s_list, only_digits)
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
func KeysFromMap(m map[string]int) []string {
	to_return := []string{}
	for k, _ := range(m) {
		to_return = append(to_return, k)
	}
	return to_return
}

// takes a slice of string-slices and returns a set of the unique elements
func SetFromLists(lists [][]string) []string {
	seen := map[string]bool{}
	for _, list := range(lists) {
		for _, str := range(list) {
			seen[str] = true
		}
	}
	to_return := []string{}
	for k, _ := range(seen) {
		to_return = append(to_return, k)
	}
	return to_return
}

func SumAndNormalizeListOfDicts(dxlist []map[string]float64) map[string]float64 {
	total := 0.0
	total_dx := make(map[string]float64)
	for _, dx := range(dxlist) {
		for t, prob := range(dx) {			
			total_dx[t] += prob
			total += prob
		}
	}
	dx := make(map[string]float64)
	for t, prob := range(total_dx) {
		dx[t] = prob / total
	}
	return dx
}
