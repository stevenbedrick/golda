package util

type Token string
type Probability float64
type Document []Token

// this is support code to make it easier to sort the results
type TokenProb struct {
	Tok Token
	Prob Probability
}

type TokenProbs []*TokenProb
type ByToken struct{ TokenProbs }
type ByProb struct{ TokenProbs }

func (t TokenProbs) Len() int { return len(t) }
func (t TokenProbs) Swap(i, j int) { t[i], t[j] = t[j], t[i] }
func (t ByToken) Less(i, j int) bool { return t.TokenProbs[i].Tok < t.TokenProbs[j].Tok }
func (t ByProb) Less(i, j int) bool { return t.TokenProbs[i].Prob < t.TokenProbs[j].Prob }

type TokenProbMap map[Token]Probability

func (tmap *TokenProbMap) TokenProbsFromMap() TokenProbs {
	to_return := TokenProbs{}
	for t, p := range(*tmap) {
		to_return = append(to_return, &TokenProb{Token(t), Probability(p)})
	}
	return to_return
}
