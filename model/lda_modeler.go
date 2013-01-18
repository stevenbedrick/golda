package model

import (
	// "fmt"
	"math/rand"
	"math"
	"lda/util"
	"sync"
	"log"
)

type LdaModeler struct {
	ntopics int
	alpha float64
	beta float64
	iterations int
	repetitions int
	seed int
	parallel bool
		
	last_model_document_topic_probability_assignments [][]float64
	last_model_topic_token_probability_assignments []map[string]float64
	last_model_token_topic_sample_assignments [][]int
	last_model_token_log_likelihood_given_topic_model float64
	last_model_sample_token_perplexity_given_topic_model float64
}

func NewModeler(ntopics int, alpha float64, beta float64, iterations int, repetitions int, seed int, parallel bool) *LdaModeler {
	return &LdaModeler{
		ntopics: ntopics, 
		alpha: alpha, 
		beta: beta, 
		iterations: iterations, 
		repetitions: repetitions, 
		seed: seed, 
		parallel: parallel,
	}
}

// this is support code to make it easier to sort the results
type TokenProb struct {
	Token string
	Probability float64
}

type TokenProbs []*TokenProb
type ByToken struct{ TokenProbs }
type ByProb struct{ TokenProbs }

func (t TokenProbs) Len() int { return len(t) }
func (t TokenProbs) Swap(i, j int) { t[i], t[j] = t[j], t[i] }
func (t ByToken) Less(i, j int) bool { return t.TokenProbs[i].Token < t.TokenProbs[j].Token }
func (t ByProb) Less(i, j int) bool { return t.TokenProbs[i].Probability < t.TokenProbs[j].Probability }

func TokenProbsFromMap(input map[string]float64) TokenProbs {
	to_return := TokenProbs{}
	for t, p := range(input) {
		to_return = append(to_return, &TokenProb{t, p})
	}
	return to_return
}

// returns a slice- one entry per topic- of token probabilities
func (l *LdaModeler) GetTopicTokenProbabilityAssignments() []map[string]float64 {
	return l.last_model_topic_token_probability_assignments
}

// stores the results of a modeling run- hopefully, using pointers like this will help cut down on reallocation.
type ModelResults struct {	
	theta_hat_ds *[][]float64
	phi_hats *[]map[string]float64
	log_likelihood float64
	perplexity float64
}

func (l *LdaModeler) Model(tokenss [][]string) {
	
	// these guys hold the results of each "run", so that they can be averaged at the end
	theta_hat_ds_samples := make([]*[][]float64, l.repetitions)
	phi_hats_samples := make([]*[]map[string]float64, l.repetitions)
	log_likelihood_samples := make([]float64, l.repetitions)
	perplexity_samples := make([]float64, l.repetitions)
	
	var initial_topicss [][]int
	
	if l.parallel && l.repetitions > 1 {
		// get a single pass to start with
		temp_res := l.model_one_pass(&tokenss, l.seed, initial_topicss)
		
		theta_hat_ds_samples[0] = temp_res.theta_hat_ds
		phi_hats_samples[0] = temp_res.phi_hats
		log_likelihood_samples[0] = temp_res.log_likelihood
		perplexity_samples[0] = temp_res.perplexity
		
		initial_topicss := l.last_model_token_topic_sample_assignments
		
		var wg sync.WaitGroup
		res_chan := make(chan *ModelResults, l.repetitions)
		
		for rep_num := 1; rep_num < l.repetitions; rep_num++ {
			wg.Add(1)
			go func(r_num int, init_tss [][]int) {
				m := NewModeler(l.ntopics, l.alpha, l.beta, l.iterations, l.repetitions, l.seed, l.parallel)
				res_chan <- m.model_one_pass(&tokenss, m.seed + r_num, init_tss)
				wg.Done()
			}(rep_num, initial_topicss)
		}
		
		wg.Wait()

		for rep_num := 1; rep_num < l.repetitions; rep_num++ {
			this_res := <- res_chan // get a result out of the channel
			theta_hat_ds_samples[rep_num] = this_res.theta_hat_ds
			phi_hats_samples[rep_num] = this_res.phi_hats
			log_likelihood_samples[rep_num] = this_res.log_likelihood
			perplexity_samples[rep_num] = this_res.perplexity			
		}
		
	} else {
		for rep_num := 0; rep_num < l.repetitions; rep_num++ {
			log.Printf("Starting repetition %d of %d.\n", rep_num + 1, l.repetitions)
			// theta_hat_ds, phi_hats, log_likelihood, perplexity := l.model_one_pass(tokenss, l.seed, initial_topicss) //temp_res 
			temp_res := l.model_one_pass(&tokenss, l.seed + rep_num, initial_topicss)
	
			theta_hat_ds_samples[rep_num] = temp_res.theta_hat_ds
			phi_hats_samples[rep_num] = temp_res.phi_hats
			log_likelihood_samples[rep_num] = temp_res.log_likelihood
			perplexity_samples[rep_num] = temp_res.perplexity
		
			if initial_topicss == nil {
				initial_topicss = l.last_model_token_topic_sample_assignments
			}		
		}
	}

	// average samples:	
	theta_hat_ds := make([][]float64, len(*theta_hat_ds_samples[0]))
	for doc_idx := 0; doc_idx < len(*theta_hat_ds_samples[0]); doc_idx++ {
		// for each doc:
		// get the appropriate theta_hat_di for this document from each repetition (should equiv to zip(*theta_hat_ds_samples) in python... )
		theta_hat_dis := make([][]float64, len(theta_hat_ds_samples))
		for rep_idx := 0; rep_idx < len(theta_hat_ds_samples); rep_idx++ {
			this_rep := *theta_hat_ds_samples[rep_idx]
			this_rep_doc := this_rep[doc_idx]
			theta_hat_dis[rep_idx] = this_rep_doc
		}
		acc_vector := make([]float64, len(theta_hat_dis[0]))
		for _, th_di := range(theta_hat_dis) {
			acc_vector, _ = util.SumVectors(acc_vector, th_di)
		}
		theta_hat_ds[doc_idx] = util.L1NormalizeVector(acc_vector)		
	}
	
	
	// phi_hats
	phi_hats := make([]map[string]float64, len(*phi_hats_samples[0]))
	for topic_idx := 0; topic_idx < len(*phi_hats_samples[0]); topic_idx++ {
		// build list of topic dicts for each topic from each rep
		this_topic := make([]map[string]float64, len(phi_hats_samples))
		for rep_idx := 0; rep_idx < len(phi_hats_samples); rep_idx++ {
			this_topic[rep_idx] = (*phi_hats_samples[rep_idx])[topic_idx]
		}
		phi_hats[topic_idx] = util.SumAndNormalizeListOfDicts(this_topic)		
	}
	
	// simple averaging for log-likelihood & perp
	log_likelihood := util.SumFloat(log_likelihood_samples) / float64(len(log_likelihood_samples))
	perplexity := util.SumFloat(perplexity_samples) / float64(len(perplexity_samples))
	
	l.last_model_document_topic_probability_assignments = theta_hat_ds
	l.last_model_topic_token_probability_assignments = phi_hats
	l.last_model_token_log_likelihood_given_topic_model = log_likelihood
	l.last_model_sample_token_perplexity_given_topic_model = perplexity
	
	log.Println("done")
}

func (l *LdaModeler) resample(r float64, values []float64, values_sum float64) int {
	r *= values_sum
	total := 0.0
	for i, p := range(values) {
		total += p
		if r <= total {
			return i
		}
	}
	return len(values) - 1.0
}

func (l *LdaModeler) model_one_pass(tokenss_ptr *[][]string, seed int, initial_topicss [][]int) *ModelResults {
	 // log.Printf("we're in %p\n", l)
	 // log.Printf("from %p: address of initial_topicss: %p\n", l, &initial_topicss)
	tokenss := *tokenss_ptr
	ntopics := l.ntopics
	ndocuments := len(tokenss)
	range_ntopics := util.PyRange(ntopics)
	range_ndocuments := util.PyRange(ndocuments)
	
	// set up a random number generator
	randomizer := rand.New(rand.NewSource(int64(len(tokenss) + seed)))
	
	// topicss is an document-token matrix where the values are topic assignments. 
	topicss := make([][]int, len(tokenss))
	
	if initial_topicss == nil {
		// set up new, randomly initialized matrix
		for i, doc := range(tokenss) {
			doc_topics := make([]int, len(doc))
			for j, _ := range(doc) { // randomly assign each token in doc to a random topic
				doc_topics[j] = randomizer.Intn(ntopics)
			}
			topicss[i] = doc_topics
		}		
	} else {
		// we were passed in a topic matrix; let's just set up our own local copy
		for i, doc := range(initial_topicss) {
			// NOTE: We *cannot* just say topicss[i] = doc- shallow copy vs. deep copy, pointers, etc. etc.- here there be possibilities for threading havoc!
			topicss[i] = make([]int, len(doc))
			copy(topicss[i], doc)
		}
	}
	
	// compute doc-topic representation counts and topic-word representation counts
	document_tokens_counts := make([]int, len(tokenss))
	for idx, doc := range(tokenss) {
		document_tokens_counts[idx] = len(doc)
	}

	document_topics_counts := make([][]int, len(tokenss))
	topic_words_counts := make([]map[string]int, ntopics) // maps in Go will, if asked for an element that doesn't exist, give the zero-value for that type (as well as an optional second return value indicating whether it was found or not) http://golang.org/doc/effective_go.html#maps
	total_topic_counts := make([]int, ntopics) // n.b.: the values in a "fresh" just-made slice are the zero-value for that type. 
	
	for i, tokens := range(tokenss) { // for each document
		topics := topicss[i]
		counts := make([]int, ntopics)
		for j, token := range(tokens) { // for each token
			topic := topics[j]
			counts[topic] += 1 // count of topic mentions in this document
			if topic_words_counts[topic] == nil {
				topic_words_counts[topic] = map[string]int{}
			}
			topic_words_counts[topic][token] += 1 // count tokens mentions for this topic
			total_topic_counts[topic] += 1 // total count of this topic
		}
		document_topics_counts[i] = counts // = append(document_topics_counts, counts)
	}

	all_keys := make([][]string, len(range_ntopics))
	for i := range(range_ntopics) {
		all_keys[i] = util.KeysFromMap(topic_words_counts[i])
	}
	all_tokens := util.SetFromLists(all_keys)
	
	// Dirichlet smoothing parameters
	alpha := l.alpha
	beta := l.beta
	W := len(all_tokens) // size of vocab: num of possible unique words in each topic
	T := ntopics // num topics
	betaW := beta * float64(W)
	alphaT := alpha * float64(T)
	// maxT := ntopics - 1
	// uniform_random_func := randomizer.Float64 // seems to be equivalent to Python's random()- uniform dist between [0.0, 1.0]
	
	// loop over all docs and all tokens, resampling topic assignments & adjusting counts
	proportional_probabilities := make([]float64, ntopics) // probability of each topic

	fixups := make([]int, ntopics) // which topics need count adjustment for current token
	for iteration := 0; iteration < l.iterations; iteration++ {
		change_count := 0
		for t_idx, tokens := range(tokenss) { // for each document
			topics := topicss[t_idx]
			document_index := range_ndocuments[t_idx]
			current_document_topics_counts := document_topics_counts[document_index]
			current_document_tokens_count := document_tokens_counts[document_index]
			n_di_minus_i := float64(current_document_tokens_count - 1)
			for token_index, token := range(tokens) { // for each token
                // Based on:
                // Griffiths TL, Steyvers M. Finding scientific topics.
                // Proceedings of the National Academy of Sciences of the United States of America. 2004;101(Suppl 1):5228-5235. 
				
				// get topic assignment for current token:
				topic := topics[token_index]

				// compute conditional probabilities for each topic,
				// the "fixups" list is an optimization to avoid branching.
				fixups[topic] = 1
				
				total_proportional_probabilities := 0.0
				for _, j := range(range_ntopics) { // for each topic
					fixup := fixups[j]

					n_wi_minus_i_j := float64(topic_words_counts[j][token] - fixup) // most of the time, fixup will be zero

					n_di_minus_i_j := float64(current_document_topics_counts[j] - fixup) // ditto
					n_dot_minus_i_j := float64(total_topic_counts[j] - fixup)
					
					// eq. 5 from above paper
					p_token_topic := (n_wi_minus_i_j + beta) / (n_dot_minus_i_j + betaW)
					p_topic_document := (n_di_minus_i_j + alpha) / (n_di_minus_i + alphaT)
					p := p_topic_document * p_token_topic
					proportional_probabilities[j] = p
					total_proportional_probabilities += p
				} // end for topics
				fixups[topic] = 0
							
				// resample current token topic, integrate the inline version of resample function
				new_topic := l.resample(randomizer.Float64(), proportional_probabilities, total_proportional_probabilities)
				
				// update assignments & counts
				if new_topic != topic {
					// update topic label for this token:
					topics[token_index] = new_topic
					
					// update total topic counts:
					total_topic_counts[topic] -= 1
					total_topic_counts[new_topic] += 1

					// update document-topic counts
					current_document_topics_counts[topic] -= 1
					current_document_topics_counts[new_topic] += 1

					topic_words_counts[topic][token] -= 1
					topic_words_counts[new_topic][token] += 1
					
					// count changes for this pass
					change_count += 1
				}
			} // end for tokens
		} // end for document
		// log.Printf("LDA - iteration %d resulted in %d changes.\n", iteration, change_count)
		if iteration % 100 == 0 {
			log.Printf("LDA - iteration %d of %d.\n", iteration, l.iterations)
		}
	} // for iterations

	// document-topic assignments (theta_hat_d_j)
	theta_hat_ds := make([][]float64, ndocuments)
	
	for document_index := 0; document_index < ndocuments; document_index++ {
		document_token_count := document_tokens_counts[document_index]
		theta_hat_d := make([]float64, ntopics)
		document_topics_count := document_topics_counts[document_index]
		if document_token_count > 0 {
			for j := 0; j < ntopics; j++ {
				p := (float64(document_topics_count[j]) + alpha) / (float64(document_token_count) + alphaT)
				theta_hat_d[j] = p
			}
		} else {
			// degenerate document with no tokens- equal prob of all topics
			// temp := float64((1.0 / ntopics) * ntopics)
			for j := range(range_ntopics) {
				theta_hat_d[j] = float64(1.0 / ntopics) //temp
			}
		}
		theta_hat_ds[document_index] = theta_hat_d
	}
	
	
	// compute topic-token assignments (phi_hat_w_j in paper)
	phi_hats := make([]map[string]float64, ntopics)
	for t := 0; t < ntopics; t++ { // for each topic
		dx := map[string]float64{}
		for token, top_tok_count := range(topic_words_counts[t]) { // for each token
			dx[token] = (float64(top_tok_count) + beta) / (float64(total_topic_counts[t]) + betaW)
		}
		phi_hats[t] = dx
	}
	
	// compute log-likelihood of tokens given topic model; Eq. 2 in Steyvers paper
	part_1, _ := math.Lgamma(float64(W) * beta) // note that Lgamma returns both the gamma, as well as a sign indicator- we don't care about the latter here
	part_2, _ := math.Lgamma(beta)
	log_likelihood := float64(T) * (part_1 - (float64(W) * part_2))
	for t := 0; t < ntopics; t++ {
		for _, w := range(all_tokens) { 
			n_t_w := topic_words_counts[t][w]
			ntw_gamma, _ := math.Lgamma(float64(n_t_w) + beta)
			log_likelihood += ntw_gamma
		}
		n_dot_t := total_topic_counts[t]
		ndt_gamma, _ := math.Lgamma(float64(n_dot_t) + betaW)
		log_likelihood -= ndt_gamma
	}
	log.Printf("LDA - log-likelihood of data given model: %0.8e\n", log_likelihood)
	
	// sum over samples:
    // See definition in:
    // Chemudugunta C, Steyvers PSM. Modeling General and Specific Aspects of Documents with a Probabilistic Topic Model.
    // In: Advances in Neural Information Processing Systems 19: Proceedings of the 2006 Conference. MIT Press; 2007.  p. 241.
    //        
    // modified implementation to add the logs of the P(tokens)'s rather than multiply
    // the P(tokens) and then take the log in order to avoid underflowing to zero.
	perplexity := 0.0
	ntokens := 0

	for doc_idx, tokens := range(tokenss) { // each document
		theta_hat_d := theta_hat_ds[doc_idx]
		for _, w := range(tokens) { // each token
			temp_phi_hat_theta_hat := make([]float64, len(range_ntopics))
			for j, z := range(range_ntopics) { // each topic
				temp_phi_hat_theta_hat[j] = phi_hats[z][w] * theta_hat_d[z]
			}
			perplexity += math.Log2(util.SumFloat(temp_phi_hat_theta_hat ))	
			ntokens += 1
		}
	}

	perplexity = math.Pow(2.0, (-perplexity / float64(ntokens) ) )
	log.Printf("LDA - mean sample token perplexity of data given model: %0.4f\n", perplexity)
	
	// save final snapshot of token-topic assignment snapshot
	if l.last_model_token_topic_sample_assignments == nil {
		l.last_model_token_topic_sample_assignments = topicss
	}
	
	// return results sample
	return &ModelResults{&theta_hat_ds, &phi_hats, log_likelihood, perplexity}
 	
}