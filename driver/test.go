package main

import (
	"fmt" 
	"regexp"
)

func main() {

	patt, _ := regexp.Compile("[\\.\\?\\!,:;\\(\\)\\'\\\"]")
	s := "this.contains.test,chars 2.4"
	z := patt.ReplaceAllString(s, " ")
	fmt.Println(z)
}