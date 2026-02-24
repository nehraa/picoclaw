package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// PaperResult holds metadata for a single academic paper.
type PaperResult struct {
	Source   string
	Title    string
	Authors  []string
	Year     string
	Abstract string
	DOI      string
	URL      string
	PDFURL   string
}

func (p *PaperResult) Format() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Title: %s\n", p.Title))
	if len(p.Authors) > 0 {
		sb.WriteString(fmt.Sprintf("Authors: %s\n", strings.Join(p.Authors, ", ")))
	}
	if p.Year != "" {
		sb.WriteString(fmt.Sprintf("Year: %s\n", p.Year))
	}
	if p.DOI != "" {
		sb.WriteString(fmt.Sprintf("DOI: %s\n", p.DOI))
	}
	if p.URL != "" {
		sb.WriteString(fmt.Sprintf("URL: %s\n", p.URL))
	}
	if p.PDFURL != "" {
		sb.WriteString(fmt.Sprintf("PDF: %s\n", p.PDFURL))
	}
	if p.Abstract != "" {
		ab := p.Abstract
		if len(ab) > 500 {
			ab = ab[:500] + "..."
		}
		sb.WriteString(fmt.Sprintf("Abstract: %s\n", ab))
	}
	sb.WriteString(fmt.Sprintf("Source: %s\n", p.Source))
	return sb.String()
}

// AcademicSearchToolOptions holds configuration for the academic search tool.
type AcademicSearchToolOptions struct {
	EmailForPolite        string
	MaxResultsPerSource   int
	SemanticScholarAPIKey string
	SpringerAPIKey        string
	IEEEAPIKey            string
	ElsevierAPIKey        string
	LensAPIKey            string
	PubMedAPIKey          string
}

// AcademicSearchTool searches academic paper sources for a given topic.
type AcademicSearchTool struct {
	opts AcademicSearchToolOptions
	fs   fileSystem
}

// NewAcademicSearchTool creates a new AcademicSearchTool.
func NewAcademicSearchTool(opts AcademicSearchToolOptions, workspace string, restrict bool) *AcademicSearchTool {
	var fs fileSystem
	if restrict {
		fs = &sandboxFs{workspace: workspace}
	} else {
		fs = &hostFs{}
	}
	if opts.MaxResultsPerSource <= 0 {
		opts.MaxResultsPerSource = 5
	}
	return &AcademicSearchTool{opts: opts, fs: fs}
}

func (t *AcademicSearchTool) Name() string {
	return "academic_search"
}

func (t *AcademicSearchTool) Description() string {
	return "Search for academic papers on a topic across multiple sources (OpenAlex, arXiv, Semantic Scholar, " +
		"Springer, IEEE, Elsevier, PLOS, PubMed, Crossref, DOAJ, DBLP, Lens.org, Unpaywall). " +
		"Returns titles, authors, abstracts, DOIs, and PDF URLs. " +
		"Optionally saves the results to a text file."
}

func (t *AcademicSearchTool) Parameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "Search topic or keywords",
			},
			"sources": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "string",
					"enum": []string{
						"openalex", "arxiv", "semantic_scholar", "unpaywall",
						"springer", "ieee", "elsevier", "plos", "pubmed",
						"crossref", "doaj", "dblp", "lens",
					},
				},
				"description": "List of sources to search. Defaults to all available (free) sources.",
			},
			"max_results": map[string]any{
				"type":        "integer",
				"description": "Maximum results per source (1-20, default 5)",
				"minimum":     1.0,
				"maximum":     20.0,
			},
			"save_to": map[string]any{
				"type":        "string",
				"description": "Optional file path to save the results as a text file",
			},
		},
		"required": []string{"query"},
	}
}

func (t *AcademicSearchTool) Execute(ctx context.Context, args map[string]any) *ToolResult {
	query, ok := args["query"].(string)
	if !ok || query == "" {
		return ErrorResult("query is required")
	}

	maxResults := t.opts.MaxResultsPerSource
	if mr, ok := args["max_results"].(float64); ok && int(mr) > 0 {
		if int(mr) <= 20 {
			maxResults = int(mr)
		}
	}

	// Determine which sources to use
	var requestedSources []string
	if srcs, ok := args["sources"].([]any); ok {
		for _, s := range srcs {
			if sv, ok := s.(string); ok {
				requestedSources = append(requestedSources, sv)
			}
		}
	}

	saveTo, _ := args["save_to"].(string)

	// Run searches
	var allResults []PaperResult
	var searchErrors []string

	type searchFunc struct {
		name string
		fn   func(context.Context, string, int) ([]PaperResult, error)
	}

	searches := t.buildSearchFuncs()

	for _, s := range searches {
		if !t.shouldSearch(s.name, requestedSources) {
			continue
		}
		results, err := s.fn(ctx, query, maxResults)
		if err != nil {
			searchErrors = append(searchErrors, fmt.Sprintf("%s: %v", s.name, err))
			continue
		}
		allResults = append(allResults, results...)
	}

	if len(allResults) == 0 && len(searchErrors) > 0 {
		return ErrorResult("all searches failed: " + strings.Join(searchErrors, "; "))
	}

	// Format output
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Academic search results for: %q\n", query))
	sb.WriteString(fmt.Sprintf("Found %d papers\n", len(allResults)))
	if len(searchErrors) > 0 {
		sb.WriteString(fmt.Sprintf("Errors: %s\n", strings.Join(searchErrors, "; ")))
	}
	sb.WriteString("\n")

	for i, paper := range allResults {
		sb.WriteString(fmt.Sprintf("--- Paper %d ---\n", i+1))
		sb.WriteString(paper.Format())
		sb.WriteString("\n")
	}

	text := sb.String()

	if saveTo != "" {
		if err := t.fs.WriteFile(saveTo, []byte(text)); err != nil {
			return ErrorResult(fmt.Sprintf("search succeeded but failed to save results: %v", err))
		}
		return &ToolResult{
			ForLLM:  fmt.Sprintf("Found %d papers for %q. Results saved to %s", len(allResults), query, saveTo),
			ForUser: text,
		}
	}

	return &ToolResult{
		ForLLM:  text,
		ForUser: text,
	}
}

type namedSearch struct {
	name string
	fn   func(context.Context, string, int) ([]PaperResult, error)
}

func (t *AcademicSearchTool) buildSearchFuncs() []namedSearch {
	searches := []namedSearch{
		{"openalex", searchOpenAlex},
		{"arxiv", searchArXiv},
		{"plos", searchPLOS},
		{"crossref", func(ctx context.Context, q string, n int) ([]PaperResult, error) {
			return searchCrossref(ctx, q, n, t.opts.EmailForPolite)
		}},
		{"doaj", searchDOAJ},
		{"dblp", searchDBLP},
		{"pubmed", func(ctx context.Context, q string, n int) ([]PaperResult, error) {
			return searchPubMed(ctx, q, n, t.opts.PubMedAPIKey)
		}},
		{"semantic_scholar", func(ctx context.Context, q string, n int) ([]PaperResult, error) {
			return searchSemanticScholar(ctx, q, n, t.opts.SemanticScholarAPIKey)
		}},
		{"springer", func(ctx context.Context, q string, n int) ([]PaperResult, error) {
			if t.opts.SpringerAPIKey == "" {
				return nil, fmt.Errorf("no API key configured")
			}
			return searchSpringer(ctx, q, n, t.opts.SpringerAPIKey)
		}},
		{"ieee", func(ctx context.Context, q string, n int) ([]PaperResult, error) {
			if t.opts.IEEEAPIKey == "" {
				return nil, fmt.Errorf("no API key configured")
			}
			return searchIEEE(ctx, q, n, t.opts.IEEEAPIKey)
		}},
		{"elsevier", func(ctx context.Context, q string, n int) ([]PaperResult, error) {
			if t.opts.ElsevierAPIKey == "" {
				return nil, fmt.Errorf("no API key configured")
			}
			return searchElsevier(ctx, q, n, t.opts.ElsevierAPIKey)
		}},
		{"lens", func(ctx context.Context, q string, n int) ([]PaperResult, error) {
			if t.opts.LensAPIKey == "" {
				return nil, fmt.Errorf("no API key configured")
			}
			return searchLens(ctx, q, n, t.opts.LensAPIKey)
		}},
	}
	return searches
}

// shouldSearch returns true if the given source should be searched.
// If requestedSources is empty, all sources are searched.
func (t *AcademicSearchTool) shouldSearch(name string, requestedSources []string) bool {
	if len(requestedSources) == 0 {
		return true
	}
	for _, s := range requestedSources {
		if s == name {
			return true
		}
	}
	return false
}

// --- OpenAlex ---

func searchOpenAlex(ctx context.Context, query string, maxResults int) ([]PaperResult, error) {
	apiURL := fmt.Sprintf(
		"https://api.openalex.org/works?search=%s&per-page=%d&select=id,title,doi,open_access,primary_location,publication_year,authorships",
		url.QueryEscape(query), maxResults,
	)
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := doHTTPRequest(req, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var data struct {
		Results []struct {
			Title           string `json:"title"`
			DOI             string `json:"doi"`
			PublicationYear int    `json:"publication_year"`
			OpenAccess      struct {
				IsOA      bool   `json:"is_oa"`
				OAURL     string `json:"oa_url"`
			} `json:"open_access"`
			PrimaryLocation struct {
				LandingPageURL string `json:"landing_page_url"`
				PDFURL         string `json:"pdf_url"`
			} `json:"primary_location"`
			Authorships []struct {
				Author struct {
					DisplayName string `json:"display_name"`
				} `json:"author"`
			} `json:"authorships"`
		} `json:"results"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []PaperResult
	for _, item := range data.Results {
		var authors []string
		for _, a := range item.Authorships {
			if a.Author.DisplayName != "" {
				authors = append(authors, a.Author.DisplayName)
			}
		}
		year := ""
		if item.PublicationYear > 0 {
			year = fmt.Sprintf("%d", item.PublicationYear)
		}
		pdfURL := item.PrimaryLocation.PDFURL
		if pdfURL == "" && item.OpenAccess.IsOA {
			pdfURL = item.OpenAccess.OAURL
		}
		results = append(results, PaperResult{
			Source:  "OpenAlex",
			Title:   item.Title,
			Authors: authors,
			Year:    year,
			DOI:     item.DOI,
			URL:     item.PrimaryLocation.LandingPageURL,
			PDFURL:  pdfURL,
		})
	}
	return results, nil
}

// --- arXiv ---

type arxivFeed struct {
	XMLName xml.Name     `xml:"feed"`
	Entries []arxivEntry `xml:"entry"`
}

type arxivEntry struct {
	ID        string `xml:"id"`
	Title     string `xml:"title"`
	Summary   string `xml:"summary"`
	Published string `xml:"published"`
	Authors   []struct {
		Name string `xml:"name"`
	} `xml:"author"`
	Links []struct {
		Href  string `xml:"href,attr"`
		Type  string `xml:"type,attr"`
		Title string `xml:"title,attr"`
		Rel   string `xml:"rel,attr"`
	} `xml:"link"`
}

func searchArXiv(ctx context.Context, query string, maxResults int) ([]PaperResult, error) {
	apiURL := fmt.Sprintf(
		"https://export.arxiv.org/api/query?search_query=all:%s&start=0&max_results=%d",
		url.QueryEscape(query), maxResults,
	)
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := doHTTPRequest(req, 20*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var feed arxivFeed
	if err := xml.Unmarshal(body, &feed); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []PaperResult
	for _, entry := range feed.Entries {
		var authors []string
		for _, a := range entry.Authors {
			authors = append(authors, a.Name)
		}
		year := ""
		if len(entry.Published) >= 4 {
			year = entry.Published[:4]
		}
		// Find PDF URL
		pdfURL := ""
		pageURL := entry.ID
		for _, link := range entry.Links {
			if link.Title == "pdf" || link.Type == "application/pdf" {
				pdfURL = link.Href
			} else if link.Rel == "alternate" {
				pageURL = link.Href
			}
		}
		// arXiv IDs look like: http://arxiv.org/abs/1234.5678v1
		// PDF is at: http://arxiv.org/pdf/1234.5678v1
		if pdfURL == "" && strings.Contains(entry.ID, "arxiv.org/abs/") {
			pdfURL = strings.Replace(entry.ID, "/abs/", "/pdf/", 1)
		}
		results = append(results, PaperResult{
			Source:   "arXiv",
			Title:    strings.TrimSpace(entry.Title),
			Authors:  authors,
			Year:     year,
			Abstract: strings.TrimSpace(entry.Summary),
			URL:      pageURL,
			PDFURL:   pdfURL,
		})
	}
	return results, nil
}

// --- Semantic Scholar ---

func searchSemanticScholar(ctx context.Context, query string, maxResults int, apiKey string) ([]PaperResult, error) {
	apiURL := fmt.Sprintf(
		"https://api.semanticscholar.org/graph/v1/paper/search?query=%s&limit=%d&fields=title,authors,year,abstract,openAccessPdf,externalIds,url",
		url.QueryEscape(query), maxResults,
	)
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	if apiKey != "" {
		req.Header.Set("x-api-key", apiKey)
	}

	resp, err := doHTTPRequest(req, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, fmt.Errorf("rate limited by Semantic Scholar")
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var data struct {
		Data []struct {
			Title   string `json:"title"`
			Year    int    `json:"year"`
			Abstract string `json:"abstract"`
			URL     string `json:"url"`
			Authors []struct {
				Name string `json:"name"`
			} `json:"authors"`
			OpenAccessPdf struct {
				URL string `json:"url"`
			} `json:"openAccessPdf"`
			ExternalIDs struct {
				DOI string `json:"DOI"`
			} `json:"externalIds"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []PaperResult
	for _, item := range data.Data {
		var authors []string
		for _, a := range item.Authors {
			authors = append(authors, a.Name)
		}
		year := ""
		if item.Year > 0 {
			year = fmt.Sprintf("%d", item.Year)
		}
		results = append(results, PaperResult{
			Source:   "Semantic Scholar",
			Title:    item.Title,
			Authors:  authors,
			Year:     year,
			Abstract: item.Abstract,
			DOI:      item.ExternalIDs.DOI,
			URL:      item.URL,
			PDFURL:   item.OpenAccessPdf.URL,
		})
	}
	return results, nil
}

// --- Springer ---

func searchSpringer(ctx context.Context, query string, maxResults int, apiKey string) ([]PaperResult, error) {
	apiURL := fmt.Sprintf(
		"https://api.springernature.com/openaccess/json?q=%s&p=%d&api_key=%s",
		url.QueryEscape(query), maxResults, url.QueryEscape(apiKey),
	)
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := doHTTPRequest(req, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncateBody(body))
	}

	var data struct {
		Records []struct {
			Title   string `json:"title"`
			DOI     string `json:"doi"`
			URL     []struct {
				Value string `json:"value"`
				Format string `json:"format"`
			} `json:"url"`
			Creators []struct {
				Creator string `json:"creator"`
			} `json:"creators"`
			PublicationDate string `json:"publicationDate"`
			Abstract        string `json:"abstract"`
		} `json:"records"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []PaperResult
	for _, item := range data.Records {
		var authors []string
		for _, c := range item.Creators {
			authors = append(authors, c.Creator)
		}
		year := ""
		if len(item.PublicationDate) >= 4 {
			year = item.PublicationDate[:4]
		}
		pageURL := ""
		pdfURL := ""
		for _, u := range item.URL {
			if u.Format == "pdf" {
				pdfURL = u.Value
			} else if pageURL == "" {
				pageURL = u.Value
			}
		}
		results = append(results, PaperResult{
			Source:   "Springer",
			Title:    item.Title,
			Authors:  authors,
			Year:     year,
			Abstract: item.Abstract,
			DOI:      item.DOI,
			URL:      pageURL,
			PDFURL:   pdfURL,
		})
	}
	return results, nil
}

// --- IEEE Xplore ---

func searchIEEE(ctx context.Context, query string, maxResults int, apiKey string) ([]PaperResult, error) {
	apiURL := fmt.Sprintf(
		"https://ieeexploreapi.ieee.org/api/v1/search/articles?querytext=%s&max_records=%d&apikey=%s",
		url.QueryEscape(query), maxResults, url.QueryEscape(apiKey),
	)
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Accept", "application/json")

	resp, err := doHTTPRequest(req, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncateBody(body))
	}

	var data struct {
		Articles []struct {
			Title       string `json:"title"`
			DOI         string `json:"doi"`
			PublicationYear string `json:"publication_year"`
			Abstract    string `json:"abstract"`
			HTMLURL     string `json:"html_url"`
			PDFLink     string `json:"pdf_url"`
			Authors     struct {
				Authors []struct {
					FullName string `json:"full_name"`
				} `json:"authors"`
			} `json:"authors"`
		} `json:"articles"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []PaperResult
	for _, item := range data.Articles {
		var authors []string
		for _, a := range item.Authors.Authors {
			authors = append(authors, a.FullName)
		}
		results = append(results, PaperResult{
			Source:   "IEEE Xplore",
			Title:    item.Title,
			Authors:  authors,
			Year:     item.PublicationYear,
			Abstract: item.Abstract,
			DOI:      item.DOI,
			URL:      item.HTMLURL,
			PDFURL:   item.PDFLink,
		})
	}
	return results, nil
}

// --- Elsevier ScienceDirect ---

func searchElsevier(ctx context.Context, query string, maxResults int, apiKey string) ([]PaperResult, error) {
	apiURL := fmt.Sprintf(
		"https://api.elsevier.com/content/search/sciencedirect?query=%s&count=%d",
		url.QueryEscape(query), maxResults,
	)
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-ELS-APIKey", apiKey)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", userAgent)

	resp, err := doHTTPRequest(req, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncateBody(body))
	}

	var data struct {
		SearchResults struct {
			Entry []struct {
				DCTitle   string `json:"dc:title"`
				DCDOI     string `json:"prism:doi"`
				DCCreator string `json:"dc:creator"`
				DCURL     string `json:"prism:url"`
				CoverDate string `json:"prism:coverDate"`
			} `json:"entry"`
		} `json:"search-results"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []PaperResult
	for _, item := range data.SearchResults.Entry {
		year := ""
		if len(item.CoverDate) >= 4 {
			year = item.CoverDate[:4]
		}
		var authors []string
		if item.DCCreator != "" {
			authors = []string{item.DCCreator}
		}
		results = append(results, PaperResult{
			Source:  "Elsevier ScienceDirect",
			Title:   item.DCTitle,
			Authors: authors,
			Year:    year,
			DOI:     item.DCDOI,
			URL:     item.DCURL,
		})
	}
	return results, nil
}

// --- PLOS ---

func searchPLOS(ctx context.Context, query string, maxResults int) ([]PaperResult, error) {
	apiURL := fmt.Sprintf(
		"https://api.plos.org/search?q=%s&rows=%d&fl=id,title,author,abstract,publication_date",
		url.QueryEscape(query), maxResults,
	)
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := doHTTPRequest(req, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncateBody(body))
	}

	var data struct {
		Response struct {
			Docs []struct {
				ID              string   `json:"id"`
				Title           string   `json:"title"`
				Author          []string `json:"author"`
				Abstract        []string `json:"abstract"`
				PublicationDate string   `json:"publication_date"`
			} `json:"docs"`
		} `json:"response"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []PaperResult
	for _, item := range data.Response.Docs {
		year := ""
		if len(item.PublicationDate) >= 4 {
			year = item.PublicationDate[:4]
		}
		abstract := ""
		if len(item.Abstract) > 0 {
			abstract = item.Abstract[0]
		}
		doi := ""
		pageURL := ""
		if item.ID != "" {
			doi = item.ID
			pageURL = "https://doi.org/" + item.ID
		}
		pdfURL := ""
		if item.ID != "" {
			// Use the generic DOI resolver instead of journal-specific URL
			pdfURL = fmt.Sprintf("https://doi.org/%s", item.ID)
		}
		results = append(results, PaperResult{
			Source:   "PLOS",
			Title:    item.Title,
			Authors:  item.Author,
			Year:     year,
			Abstract: abstract,
			DOI:      doi,
			URL:      pageURL,
			PDFURL:   pdfURL,
		})
	}
	return results, nil
}

// --- PubMed Central (PMC) ---

func searchPubMed(ctx context.Context, query string, maxResults int, apiKey string) ([]PaperResult, error) {
	// Step 1: esearch to get IDs
	searchURL := fmt.Sprintf(
		"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&term=%s&retmax=%d&retmode=json",
		url.QueryEscape(query), maxResults,
	)
	if apiKey != "" {
		searchURL += "&api_key=" + url.QueryEscape(apiKey)
	}

	req, err := http.NewRequestWithContext(ctx, "GET", searchURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := doHTTPRequest(req, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var searchResult struct {
		ESearchResult struct {
			IDList []string `json:"idlist"`
		} `json:"esearchresult"`
	}
	if err := json.Unmarshal(body, &searchResult); err != nil {
		return nil, fmt.Errorf("parse esearch error: %w", err)
	}

	ids := searchResult.ESearchResult.IDList
	if len(ids) == 0 {
		return nil, nil
	}
	if len(ids) > maxResults {
		ids = ids[:maxResults]
	}

	// Step 2: esummary to get metadata
	summaryURL := fmt.Sprintf(
		"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pmc&id=%s&retmode=json",
		strings.Join(ids, ","),
	)
	if apiKey != "" {
		summaryURL += "&api_key=" + url.QueryEscape(apiKey)
	}

	req2, err := http.NewRequestWithContext(ctx, "GET", summaryURL, nil)
	if err != nil {
		return nil, err
	}
	req2.Header.Set("User-Agent", userAgent)

	resp2, err := doHTTPRequest(req2, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp2.Body.Close()

	body2, err := io.ReadAll(resp2.Body)
	if err != nil {
		return nil, err
	}

	var summaryResult struct {
		Result map[string]json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(body2, &summaryResult); err != nil {
		return nil, fmt.Errorf("parse esummary error: %w", err)
	}

	var results []PaperResult
	for _, id := range ids {
		raw, ok := summaryResult.Result[id]
		if !ok {
			continue
		}
		var doc struct {
			Title   string `json:"title"`
			PubDate string `json:"pubdate"`
			Authors []struct {
				Name string `json:"name"`
			} `json:"authors"`
			DOI string `json:"elocationid"`
		}
		if err := json.Unmarshal(raw, &doc); err != nil {
			continue
		}
		var authors []string
		for _, a := range doc.Authors {
			authors = append(authors, a.Name)
		}
		year := ""
		if len(doc.PubDate) >= 4 {
			year = doc.PubDate[:4]
		}
		pageURL := fmt.Sprintf("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC%s/", id)
		pdfURL := fmt.Sprintf("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC%s/pdf/", id)
		results = append(results, PaperResult{
			Source:  "PubMed Central",
			Title:   doc.Title,
			Authors: authors,
			Year:    year,
			DOI:     doc.DOI,
			URL:     pageURL,
			PDFURL:  pdfURL,
		})
	}
	return results, nil
}

// --- Crossref ---

func searchCrossref(ctx context.Context, query string, maxResults int, email string) ([]PaperResult, error) {
	apiURL := fmt.Sprintf(
		"https://api.crossref.org/works?query=%s&rows=%d&select=title,author,published,DOI,link,abstract",
		url.QueryEscape(query), maxResults,
	)
	if email != "" {
		apiURL += "&mailto=" + url.QueryEscape(email)
	}

	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := doHTTPRequest(req, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var data struct {
		Message struct {
			Items []struct {
				Title    []string `json:"title"`
				DOI      string   `json:"DOI"`
				Abstract string   `json:"abstract"`
				Published struct {
					DateParts [][]int `json:"date-parts"`
				} `json:"published"`
				Author []struct {
					Given  string `json:"given"`
					Family string `json:"family"`
				} `json:"author"`
				Link []struct {
					URL         string `json:"URL"`
					ContentType string `json:"content-type"`
				} `json:"link"`
			} `json:"items"`
		} `json:"message"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []PaperResult
	for _, item := range data.Message.Items {
		title := ""
		if len(item.Title) > 0 {
			title = item.Title[0]
		}
		var authors []string
		for _, a := range item.Author {
			name := strings.TrimSpace(a.Given + " " + a.Family)
			if name != "" {
				authors = append(authors, name)
			}
		}
		year := ""
		if len(item.Published.DateParts) > 0 && len(item.Published.DateParts[0]) > 0 {
			year = fmt.Sprintf("%d", item.Published.DateParts[0][0])
		}
		doi := item.DOI
		pageURL := ""
		pdfURL := ""
		if doi != "" {
			pageURL = "https://doi.org/" + doi
		}
		for _, link := range item.Link {
			if link.ContentType == "application/pdf" {
				pdfURL = link.URL
			}
		}
		results = append(results, PaperResult{
			Source:   "Crossref",
			Title:    title,
			Authors:  authors,
			Year:     year,
			Abstract: item.Abstract,
			DOI:      doi,
			URL:      pageURL,
			PDFURL:   pdfURL,
		})
	}
	return results, nil
}

// --- DOAJ ---

func searchDOAJ(ctx context.Context, query string, maxResults int) ([]PaperResult, error) {
	apiURL := fmt.Sprintf(
		"https://doaj.org/api/search/articles/%s?pageSize=%d",
		url.PathEscape(query), maxResults,
	)
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := doHTTPRequest(req, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncateBody(body))
	}

	var data struct {
		Results []struct {
			Bibjson struct {
				Title   string `json:"title"`
				Abstract string `json:"abstract"`
				Year    string `json:"year"`
				Author  []struct {
					Name string `json:"name"`
				} `json:"author"`
				Identifier []struct {
					Type string `json:"type"`
					ID   string `json:"id"`
				} `json:"identifier"`
				Link []struct {
					URL  string `json:"url"`
					Type string `json:"type"`
				} `json:"link"`
			} `json:"bibjson"`
		} `json:"results"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []PaperResult
	for _, item := range data.Results {
		bib := item.Bibjson
		var authors []string
		for _, a := range bib.Author {
			authors = append(authors, a.Name)
		}
		doi := ""
		for _, id := range bib.Identifier {
			if id.Type == "doi" {
				doi = id.ID
			}
		}
		pageURL := ""
		pdfURL := ""
		for _, link := range bib.Link {
			if link.Type == "fulltext" {
				pageURL = link.URL
			} else if link.Type == "pdf" {
				pdfURL = link.URL
			}
		}
		if pageURL == "" && doi != "" {
			pageURL = "https://doi.org/" + doi
		}
		results = append(results, PaperResult{
			Source:   "DOAJ",
			Title:    bib.Title,
			Authors:  authors,
			Year:     bib.Year,
			Abstract: bib.Abstract,
			DOI:      doi,
			URL:      pageURL,
			PDFURL:   pdfURL,
		})
	}
	return results, nil
}

// --- DBLP ---

func searchDBLP(ctx context.Context, query string, maxResults int) ([]PaperResult, error) {
	apiURL := fmt.Sprintf(
		"https://dblp.org/search/publ/api?q=%s&format=json&h=%d",
		url.QueryEscape(query), maxResults,
	)
	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := doHTTPRequest(req, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var data struct {
		Result struct {
			Hits struct {
				Hit []struct {
					Info struct {
						Title   string `json:"title"`
						Year    string `json:"year"`
						URL     string `json:"url"`
						DOI     string `json:"doi"`
						Authors struct {
							Author any `json:"author"` // can be string or []string
						} `json:"authors"`
					} `json:"info"`
				} `json:"hit"`
			} `json:"hits"`
		} `json:"result"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []PaperResult
	for _, hit := range data.Result.Hits.Hit {
		info := hit.Info
		var authors []string
		switch v := info.Authors.Author.(type) {
		case string:
			authors = []string{v}
		case []any:
			for _, a := range v {
				if s, ok := a.(string); ok {
					authors = append(authors, s)
				}
			}
		}
		results = append(results, PaperResult{
			Source:  "DBLP",
			Title:   info.Title,
			Authors: authors,
			Year:    info.Year,
			DOI:     info.DOI,
			URL:     info.URL,
		})
	}
	return results, nil
}

// --- Lens.org ---

func searchLens(ctx context.Context, query string, maxResults int, apiKey string) ([]PaperResult, error) {
	payload := map[string]any{
		"query": map[string]any{
			"match": map[string]any{
				"title": query,
			},
		},
		"size":    maxResults,
		"include": []string{"title", "authors", "year_published", "abstract", "doi", "open_access", "external_ids", "scholarly_citations_count"},
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.lens.org/scholarly/search", bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", userAgent)

	resp, err := doHTTPRequest(req, 15*time.Second)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncateBody(body))
	}

	var data struct {
		Data []struct {
			Title          string `json:"title"`
			YearPublished  int    `json:"year_published"`
			Abstract       string `json:"abstract"`
			DOI            string `json:"doi"`
			Authors        []struct {
				DisplayName string `json:"display_name"`
			} `json:"authors"`
			OpenAccess struct {
				IsOA    bool   `json:"is_oa"`
				License string `json:"license"`
			} `json:"open_access"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var results []PaperResult
	for _, item := range data.Data {
		var authors []string
		for _, a := range item.Authors {
			authors = append(authors, a.DisplayName)
		}
		year := ""
		if item.YearPublished > 0 {
			year = fmt.Sprintf("%d", item.YearPublished)
		}
		pageURL := ""
		if item.DOI != "" {
			pageURL = "https://doi.org/" + item.DOI
		}
		results = append(results, PaperResult{
			Source:   "Lens.org",
			Title:    item.Title,
			Authors:  authors,
			Year:     year,
			Abstract: item.Abstract,
			DOI:      item.DOI,
			URL:      pageURL,
		})
	}
	return results, nil
}

// --- Unpaywall (per-DOI open access lookup) ---

// UnpaywallLookup checks whether an open-access version of a paper is available by DOI.
// It is used by AcademicFetchPaperTool when a DOI is provided without a direct URL.
func unpaywallLookup(ctx context.Context, doi, email string) (pdfURL string, err error) {
	apiURL := fmt.Sprintf("https://api.unpaywall.org/v2/%s?email=%s",
		url.PathEscape(doi), url.QueryEscape(email))

	req, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := doHTTPRequest(req, 10*time.Second)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	var data struct {
		BestOALocation struct {
			URLPDF string `json:"url_for_pdf"`
			URL    string `json:"url"`
		} `json:"best_oa_location"`
		IsOA bool `json:"is_oa"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}
	if !data.IsOA {
		return "", nil
	}
	if data.BestOALocation.URLPDF != "" {
		return data.BestOALocation.URLPDF, nil
	}
	return data.BestOALocation.URL, nil
}

// --- AcademicFetchPaperTool ---

// AcademicFetchPaperTool downloads a paper (PDF or HTML) from a URL or DOI and saves it to a file.
type AcademicFetchPaperTool struct {
	emailForPolite string
	fs             fileSystem
}

// NewAcademicFetchPaperTool creates a new AcademicFetchPaperTool.
func NewAcademicFetchPaperTool(emailForPolite, workspace string, restrict bool) *AcademicFetchPaperTool {
	var fs fileSystem
	if restrict {
		fs = &sandboxFs{workspace: workspace}
	} else {
		fs = &hostFs{}
	}
	return &AcademicFetchPaperTool{emailForPolite: emailForPolite, fs: fs}
}

func (t *AcademicFetchPaperTool) Name() string {
	return "academic_fetch_paper"
}

func (t *AcademicFetchPaperTool) Description() string {
	return "Download a specific academic paper by URL or DOI and save it to a file. " +
		"PDFs are saved as binary .pdf files. HTML/text pages are saved as .txt files with extracted text. " +
		"Supports Unpaywall to find open-access PDFs by DOI."
}

func (t *AcademicFetchPaperTool) Parameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"url": map[string]any{
				"type":        "string",
				"description": "Direct URL of the paper (PDF or HTML page)",
			},
			"doi": map[string]any{
				"type":        "string",
				"description": "DOI of the paper (e.g. '10.1038/nature12345'). Used to find an open-access PDF via Unpaywall if no url is given.",
			},
			"save_to": map[string]any{
				"type":        "string",
				"description": "File path to save the paper. Use .pdf extension for PDFs or .txt for text.",
			},
		},
		"required": []string{"save_to"},
	}
}

func (t *AcademicFetchPaperTool) Execute(ctx context.Context, args map[string]any) *ToolResult {
	saveTo, ok := args["save_to"].(string)
	if !ok || saveTo == "" {
		return ErrorResult("save_to is required")
	}

	paperURL, _ := args["url"].(string)
	doi, _ := args["doi"].(string)

	// If no URL given but DOI provided, try Unpaywall
	if paperURL == "" && doi != "" {
		if t.emailForPolite == "" {
			return ErrorResult("email_for_polite must be set in academic tools config to use DOI/Unpaywall lookup")
		}
		oaURL, err := unpaywallLookup(ctx, doi, t.emailForPolite)
		if err != nil {
			return ErrorResult(fmt.Sprintf("Unpaywall lookup failed for DOI %s: %v", doi, err))
		}
		if oaURL == "" {
			return ErrorResult(fmt.Sprintf("no open-access version found for DOI %s", doi))
		}
		paperURL = oaURL
	}

	if paperURL == "" {
		return ErrorResult("either url or doi must be provided")
	}

	// Validate URL
	parsed, err := url.Parse(paperURL)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return ErrorResult("only http/https URLs are supported")
	}

	req, err := http.NewRequestWithContext(ctx, "GET", paperURL, nil)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to create request: %v", err))
	}
	req.Header.Set("User-Agent", userAgent)

	client := &http.Client{
		Timeout: 120 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return fmt.Errorf("stopped after 5 redirects")
			}
			return nil
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return ErrorResult(fmt.Sprintf("request failed: %v", err))
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return ErrorResult(fmt.Sprintf("HTTP %d when fetching %s", resp.StatusCode, paperURL))
	}

	contentType := resp.Header.Get("Content-Type")
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to read response: %v", err))
	}

	isPDF := strings.Contains(contentType, "application/pdf") ||
		strings.HasSuffix(strings.ToLower(paperURL), ".pdf") ||
		(len(body) >= 4 && string(body[:4]) == "%PDF")

	var saveData []byte
	var fileType string

	if isPDF {
		saveData = body
		fileType = "PDF"
	} else {
		// Extract text for HTML/text content
		var text string
		if strings.Contains(contentType, "text/html") {
			wft := &WebFetchTool{}
			text = wft.extractText(string(body))
		} else {
			text = string(body)
		}
		// Add source metadata header
		header := fmt.Sprintf("Source: %s\nFetched: %s\n\n", paperURL, time.Now().UTC().Format(time.RFC3339))
		text = header + text
		saveData = []byte(text)
		fileType = "text"
	}

	if err := t.fs.WriteFile(saveTo, saveData); err != nil {
		return ErrorResult(fmt.Sprintf("failed to save file: %v", err))
	}

	msg := fmt.Sprintf("Paper saved as %s (%d bytes) to %s", fileType, len(saveData), saveTo)
	return &ToolResult{
		ForLLM:  msg,
		ForUser: msg,
	}
}

// --- helpers ---

func doHTTPRequest(req *http.Request, timeout time.Duration) (*http.Response, error) {
	client := &http.Client{Timeout: timeout}
	return client.Do(req)
}

func truncateBody(body []byte) string {
	if len(body) > 200 {
		return string(body[:200]) + "..."
	}
	return string(body)
}
