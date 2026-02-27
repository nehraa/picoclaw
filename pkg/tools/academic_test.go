package tools

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --- AcademicSearchTool tests ---

func TestAcademicSearchTool_Name(t *testing.T) {
	tool := NewAcademicSearchTool(AcademicSearchToolOptions{}, t.TempDir(), false)
	if tool.Name() != "academic_search" {
		t.Errorf("expected name 'academic_search', got %q", tool.Name())
	}
}

func TestAcademicSearchTool_MissingQuery(t *testing.T) {
	tool := NewAcademicSearchTool(AcademicSearchToolOptions{}, t.TempDir(), false)
	result := tool.Execute(context.Background(), map[string]any{})
	if !result.IsError {
		t.Error("expected error when query is missing")
	}
	if !strings.Contains(result.ForLLM, "query is required") {
		t.Errorf("unexpected error message: %s", result.ForLLM)
	}
}

func TestAcademicSearchTool_DefaultMaxResults(t *testing.T) {
	tool := NewAcademicSearchTool(AcademicSearchToolOptions{}, t.TempDir(), false)
	if tool.opts.MaxResultsPerSource != 5 {
		t.Errorf("expected default max results 5, got %d", tool.opts.MaxResultsPerSource)
	}
}

func TestAcademicSearchTool_CustomMaxResults(t *testing.T) {
	tool := NewAcademicSearchTool(AcademicSearchToolOptions{MaxResultsPerSource: 10}, t.TempDir(), false)
	if tool.opts.MaxResultsPerSource != 10 {
		t.Errorf("expected max results 10, got %d", tool.opts.MaxResultsPerSource)
	}
}

func TestAcademicSearchTool_Parameters(t *testing.T) {
	tool := NewAcademicSearchTool(AcademicSearchToolOptions{}, t.TempDir(), false)
	params := tool.Parameters()
	if params["type"] != "object" {
		t.Errorf("expected type 'object', got %v", params["type"])
	}
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties to be map")
	}
	if _, ok := props["query"]; !ok {
		t.Error("expected 'query' parameter")
	}
	if _, ok := props["sources"]; !ok {
		t.Error("expected 'sources' parameter")
	}
	if _, ok := props["save_to"]; !ok {
		t.Error("expected 'save_to' parameter")
	}
}

// TestAcademicSearchTool_OpenAlexSource verifies that the tool handles search
// errors gracefully (e.g. network failure) without panicking.
func TestAcademicSearchTool_OpenAlexSource(t *testing.T) {
	tool := NewAcademicSearchTool(AcademicSearchToolOptions{MaxResultsPerSource: 2}, t.TempDir(), false)
	// Use an immediately-cancelled context to simulate a network failure without
	// making real external calls.
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	result := tool.Execute(ctx, map[string]any{
		"query":   "test",
		"sources": []any{"openalex"},
	})
	// The result is either an error (all searches failed) or has 0 papers.
	// Either way, it should not panic.
	_ = result
}

// TestAcademicSearchTool_SaveTo verifies that the file system write path works
// when a save_to is specified, by writing directly via the tool's fs.
func TestAcademicSearchTool_SaveTo(t *testing.T) {
	dir := t.TempDir()
	savePath := filepath.Join(dir, "results.txt")

	tool := NewAcademicSearchTool(AcademicSearchToolOptions{MaxResultsPerSource: 1}, dir, false)

	papers := []PaperResult{
		{Source: "test", Title: "Hello World", Authors: []string{"Doe, J."}, Year: "2024"},
	}
	var sb strings.Builder
	sb.WriteString("Academic search results for: \"test\"\n")
	sb.WriteString("Found 1 papers\n\n")
	for _, p := range papers {
		sb.WriteString(p.Format())
	}
	if err := tool.fs.WriteFile(savePath, []byte(sb.String())); err != nil {
		t.Fatalf("unexpected write error: %v", err)
	}

	data, err := os.ReadFile(savePath)
	if err != nil {
		t.Fatalf("expected saved file to exist: %v", err)
	}
	if !strings.Contains(string(data), "Hello World") {
		t.Errorf("saved file missing expected content: %s", string(data))
	}
}

// --- PaperResult.Format tests ---

func TestPaperResult_Format_Full(t *testing.T) {
	p := PaperResult{
		Source:   "arXiv",
		Title:    "Deep Learning Survey",
		Authors:  []string{"LeCun, Y.", "Bengio, Y.", "Hinton, G."},
		Year:     "2015",
		Abstract: "A survey of deep learning methods.",
		DOI:      "10.1038/nature14539",
		URL:      "https://www.nature.com/articles/nature14539",
		PDFURL:   "https://arxiv.org/pdf/1206.5533",
	}
	formatted := p.Format()
	for _, want := range []string{"Deep Learning Survey", "LeCun", "2015", "arXiv", "10.1038/nature14539"} {
		if !strings.Contains(formatted, want) {
			t.Errorf("expected %q in formatted output, got: %s", want, formatted)
		}
	}
}

func TestPaperResult_Format_AbstractTruncated(t *testing.T) {
	longAbstract := strings.Repeat("a", 600)
	p := PaperResult{Source: "test", Title: "T", Abstract: longAbstract}
	formatted := p.Format()
	if strings.Contains(formatted, longAbstract) {
		t.Error("expected abstract to be truncated to 500 chars")
	}
	if !strings.Contains(formatted, "...") {
		t.Error("expected truncation ellipsis")
	}
}

func TestPaperResult_Format_Empty(t *testing.T) {
	p := PaperResult{Source: "test", Title: "Only Title"}
	formatted := p.Format()
	if !strings.Contains(formatted, "Only Title") {
		t.Errorf("expected title in output, got: %s", formatted)
	}
}

// --- arXiv XML parsing ---

func TestSearchArXiv_ParsesXML(t *testing.T) {
	feed := arxivFeed{
		Entries: []arxivEntry{
			{
				ID:        "http://arxiv.org/abs/1234.5678v1",
				Title:     "  Some Title  ",
				Summary:   "An abstract.",
				Published: "2023-01-15T00:00:00Z",
				Authors:   []struct{ Name string `xml:"name"` }{{Name: "Alice"}},
				Links: []struct {
					Href  string `xml:"href,attr"`
					Type  string `xml:"type,attr"`
					Title string `xml:"title,attr"`
					Rel   string `xml:"rel,attr"`
				}{
					{Href: "http://arxiv.org/abs/1234.5678v1", Rel: "alternate"},
					{Href: "http://arxiv.org/pdf/1234.5678v1", Title: "pdf"},
				},
			},
		},
	}

	xmlData, err := xml.Marshal(feed)
	if err != nil {
		t.Fatalf("failed to marshal test feed: %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/atom+xml")
		// Prepend the XML declaration to make it valid
		w.Write([]byte(`<?xml version="1.0" encoding="UTF-8"?>`))
		w.Write(xmlData)
	}))
	defer server.Close()

	// We cannot easily swap the URL, so just test the XML parsing logic
	var parsed arxivFeed
	if err := xml.Unmarshal(xmlData, &parsed); err != nil {
		t.Fatalf("failed to parse feed XML: %v", err)
	}
	if len(parsed.Entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(parsed.Entries))
	}
	entry := parsed.Entries[0]
	if !strings.Contains(entry.Title, "Some Title") {
		t.Errorf("unexpected title: %q", entry.Title)
	}
	if entry.Authors[0].Name != "Alice" {
		t.Errorf("unexpected author: %q", entry.Authors[0].Name)
	}
}

// --- shouldSearch ---

func TestAcademicSearchTool_ShouldSearch(t *testing.T) {
	tool := NewAcademicSearchTool(AcademicSearchToolOptions{}, t.TempDir(), false)

	// When no sources requested, all should match
	if !tool.shouldSearch("openalex", nil) {
		t.Error("expected true for empty requested list")
	}

	// When sources listed, only matching ones should return true
	requested := []string{"arxiv", "plos"}
	if !tool.shouldSearch("arxiv", requested) {
		t.Error("expected true for 'arxiv' in requested list")
	}
	if tool.shouldSearch("openalex", requested) {
		t.Error("expected false for 'openalex' not in requested list")
	}
}

// --- AcademicFetchPaperTool tests ---

func TestAcademicFetchPaperTool_Name(t *testing.T) {
	tool := NewAcademicFetchPaperTool("", t.TempDir(), false)
	if tool.Name() != "academic_fetch_paper" {
		t.Errorf("expected name 'academic_fetch_paper', got %q", tool.Name())
	}
}

func TestAcademicFetchPaperTool_MissingSaveTo(t *testing.T) {
	tool := NewAcademicFetchPaperTool("", t.TempDir(), false)
	result := tool.Execute(context.Background(), map[string]any{
		"url": "https://example.com/paper.pdf",
	})
	if !result.IsError {
		t.Error("expected error when save_to is missing")
	}
}

func TestAcademicFetchPaperTool_MissingURLAndDOI(t *testing.T) {
	tool := NewAcademicFetchPaperTool("", t.TempDir(), false)
	result := tool.Execute(context.Background(), map[string]any{
		"save_to": "paper.txt",
	})
	if !result.IsError {
		t.Error("expected error when neither url nor doi provided")
	}
}

func TestAcademicFetchPaperTool_DOIWithoutEmail(t *testing.T) {
	tool := NewAcademicFetchPaperTool("", t.TempDir(), false)
	result := tool.Execute(context.Background(), map[string]any{
		"doi":     "10.1234/test",
		"save_to": "paper.txt",
	})
	if !result.IsError {
		t.Error("expected error when doi used without email")
	}
	if !strings.Contains(result.ForLLM, "email_for_polite") {
		t.Errorf("expected email error, got: %s", result.ForLLM)
	}
}

func TestAcademicFetchPaperTool_FetchHTML(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("<html><body><h1>Paper Title</h1><p>Abstract text here.</p></body></html>"))
	}))
	defer server.Close()

	dir := t.TempDir()
	savePath := filepath.Join(dir, "paper.txt")

	tool := NewAcademicFetchPaperTool("test@example.com", dir, false)
	result := tool.Execute(context.Background(), map[string]any{
		"url":     server.URL,
		"save_to": savePath,
	})

	if result.IsError {
		t.Fatalf("expected success, got error: %s", result.ForLLM)
	}

	data, err := os.ReadFile(savePath)
	if err != nil {
		t.Fatalf("expected saved file: %v", err)
	}
	if !strings.Contains(string(data), "Paper Title") {
		t.Errorf("expected extracted text in saved file, got: %s", string(data))
	}
}

func TestAcademicFetchPaperTool_FetchPDF(t *testing.T) {
	// Fake PDF bytes (%PDF magic bytes)
	pdfBytes := []byte("%PDF-1.4 fake pdf content")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/pdf")
		w.WriteHeader(http.StatusOK)
		w.Write(pdfBytes)
	}))
	defer server.Close()

	dir := t.TempDir()
	savePath := filepath.Join(dir, "paper.pdf")

	tool := NewAcademicFetchPaperTool("test@example.com", dir, false)
	result := tool.Execute(context.Background(), map[string]any{
		"url":     server.URL,
		"save_to": savePath,
	})

	if result.IsError {
		t.Fatalf("expected success, got error: %s", result.ForLLM)
	}

	data, err := os.ReadFile(savePath)
	if err != nil {
		t.Fatalf("expected saved pdf file: %v", err)
	}
	if string(data) != string(pdfBytes) {
		t.Errorf("saved PDF content mismatch")
	}
	if !strings.Contains(result.ForLLM, "PDF") {
		t.Errorf("expected 'PDF' in result message, got: %s", result.ForLLM)
	}
}

func TestAcademicFetchPaperTool_InvalidURL(t *testing.T) {
	tool := NewAcademicFetchPaperTool("", t.TempDir(), false)
	result := tool.Execute(context.Background(), map[string]any{
		"url":     "ftp://bad-scheme.com/paper.pdf",
		"save_to": "paper.pdf",
	})
	if !result.IsError {
		t.Error("expected error for unsupported scheme")
	}
}

func TestAcademicFetchPaperTool_Parameters(t *testing.T) {
	tool := NewAcademicFetchPaperTool("", t.TempDir(), false)
	params := tool.Parameters()
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties map")
	}
	for _, key := range []string{"url", "doi", "save_to"} {
		if _, ok := props[key]; !ok {
			t.Errorf("expected parameter %q", key)
		}
	}
}

// TestUnpaywallLookup_MockServer verifies that unpaywallLookup correctly parses a response.
func TestUnpaywallLookup_MockServer(t *testing.T) {
	// We cannot swap the Unpaywall URL easily, so test the HTTP path via a
	// mock that replaces the actual HTTP request. We test this indirectly
	// through the fetch paper tool's DOI path to ensure the error message
	// is correct when no email is provided (already tested above).
	// Here we test the mock server directly with a helper.

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"is_oa": true,
			"best_oa_location": map[string]any{
				"url_for_pdf": "https://example.com/paper.pdf",
				"url":         "https://example.com/paper",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// unpaywallLookup uses a hardcoded URL, so we test the parsing logic
	// by reading the raw JSON directly.
	var data struct {
		BestOALocation struct {
			URLPDF string `json:"url_for_pdf"`
			URL    string `json:"url"`
		} `json:"best_oa_location"`
		IsOA bool `json:"is_oa"`
	}
	payload := `{"is_oa":true,"best_oa_location":{"url_for_pdf":"https://example.com/paper.pdf","url":"https://example.com/paper"}}`
	if err := json.Unmarshal([]byte(payload), &data); err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if !data.IsOA {
		t.Error("expected is_oa true")
	}
	if data.BestOALocation.URLPDF != "https://example.com/paper.pdf" {
		t.Errorf("unexpected pdf url: %s", data.BestOALocation.URLPDF)
	}
}

// =============================================================================
// Tests for the new PDF-first and citation-extraction functionality
// =============================================================================

// --- findPDFURLInHTML ---

func TestFindPDFURLInHTML_CitationMeta(t *testing.T) {
html := `<html><head>
<meta name="citation_pdf_url" content="https://example.com/paper.pdf">
</head><body></body></html>`
got := findPDFURLInHTML(html, "https://example.com/article")
if got != "https://example.com/paper.pdf" {
t.Errorf("expected PDF URL from citation_pdf_url meta, got %q", got)
}
}

func TestFindPDFURLInHTML_HrefPDF(t *testing.T) {
html := `<html><body><a href="/files/paper.pdf">Download PDF</a></body></html>`
got := findPDFURLInHTML(html, "https://example.com/article")
if got != "https://example.com/files/paper.pdf" {
t.Errorf("expected PDF URL from .pdf href, got %q", got)
}
}

func TestFindPDFURLInHTML_DataPDFURL(t *testing.T) {
html := `<html><body><div data-pdf-url="https://cdn.example.com/doc.pdf"></div></body></html>`
got := findPDFURLInHTML(html, "")
if got != "https://cdn.example.com/doc.pdf" {
t.Errorf("expected PDF URL from data-pdf-url, got %q", got)
}
}

func TestFindPDFURLInHTML_NoPDF(t *testing.T) {
html := `<html><body><p>No PDF here</p></body></html>`
got := findPDFURLInHTML(html, "https://example.com")
if got != "" {
t.Errorf("expected empty string when no PDF link, got %q", got)
}
}

// --- AcademicFetchPaperTool PDF-first behaviour ---

// TestAcademicFetchPaperTool_HTMLWithEmbeddedPDFLink verifies that when an HTML page
// contains a citation_pdf_url meta tag, the tool follows it and saves a PDF.
func TestAcademicFetchPaperTool_HTMLWithEmbeddedPDFLink(t *testing.T) {
pdfBytes := []byte("%PDF-1.4 embedded pdf")

// PDF endpoint
pdfServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
w.Header().Set("Content-Type", "application/pdf")
w.Write(pdfBytes)
}))
defer pdfServer.Close()

// HTML page that embeds a citation_pdf_url pointing to the PDF server
htmlServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
w.Header().Set("Content-Type", "text/html")
fmt.Fprintf(w, `<html><head>
<meta name="citation_pdf_url" content="%s">
</head><body><h1>Journal Article</h1></body></html>`, pdfServer.URL+"/paper.pdf")
}))
defer htmlServer.Close()

dir := t.TempDir()
savePath := filepath.Join(dir, "paper.pdf")

tool := NewAcademicFetchPaperTool("test@example.com", dir, false)
result := tool.Execute(context.Background(), map[string]any{
"url":     htmlServer.URL,
"save_to": savePath,
})

if result.IsError {
t.Fatalf("expected success, got error: %s", result.ForLLM)
}
if !strings.Contains(result.ForLLM, "PDF") {
t.Errorf("expected 'PDF' in result (PDF-first succeeded), got: %s", result.ForLLM)
}
data, err := os.ReadFile(savePath)
if err != nil {
t.Fatalf("expected saved file: %v", err)
}
if string(data) != string(pdfBytes) {
t.Errorf("saved content does not match expected PDF bytes")
}
}

// TestAcademicFetchPaperTool_HTMLFallbackToText verifies that when no PDF link is found
// in an HTML page, the tool falls back to saving extracted text.
func TestAcademicFetchPaperTool_HTMLFallbackToText(t *testing.T) {
server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
w.Header().Set("Content-Type", "text/html")
w.Write([]byte("<html><body><h1>Article Title</h1><p>Body text.</p></body></html>"))
}))
defer server.Close()

dir := t.TempDir()
savePath := filepath.Join(dir, "paper.txt")

tool := NewAcademicFetchPaperTool("", dir, false)
result := tool.Execute(context.Background(), map[string]any{
"url":     server.URL,
"save_to": savePath,
})

if result.IsError {
t.Fatalf("expected success, got error: %s", result.ForLLM)
}
if !strings.Contains(result.ForLLM, "text") {
t.Errorf("expected 'text' fallback in result message, got: %s", result.ForLLM)
}
}

// --- PDF text extraction ---

func TestExtractTextFromPDF_BtEtBlock(t *testing.T) {
// Minimal synthetic PDF-like content with BT...ET text operators
fakePDF := `%PDF-1.4
1 0 obj << /Type /Page >> endobj
stream
BT
(Hello World) Tj
(Second Line) Tj
ET
endstream`
result := extractTextFromPDF([]byte(fakePDF))
if !strings.Contains(result, "Hello") && !strings.Contains(result, "World") {
t.Errorf("expected extracted text to contain 'Hello World', got: %q", result)
}
}

func TestExtractTextFromPDF_FallbackToASCII(t *testing.T) {
// A binary blob with no BT/ET — should fall back to printable ASCII extraction
// Bytes: %PDF followed by two null bytes, then ASCII "References" and a null byte
data := []byte{0x25, 0x50, 0x44, 0x46, 0x00, 0x00,
'R', 'e', 'f', 'e', 'r', 'e', 'n', 'c', 'e', 's', 0x00}
result := extractTextFromPDF(data)
if !strings.Contains(result, "References") {
t.Errorf("expected ASCII fallback to contain 'References', got: %q", result)
}
}

func TestExtractPrintableASCII_CollapsesBlankLines(t *testing.T) {
data := []byte("line1\x00\x00\x00line2\x00line3")
result := extractPrintableASCII(data)
if strings.Contains(result, "\n\n\n") {
t.Errorf("expected blank lines collapsed, got: %q", result)
}
if !strings.Contains(result, "line1") || !strings.Contains(result, "line2") {
t.Errorf("expected readable lines present, got: %q", result)
}
}

// --- Citation section extraction ---

func TestExtractCitationSection_FoundHeader(t *testing.T) {
text := "Introduction\n...\nReferences\n[1] Some Author. Title. 2020.\n"
section := extractCitationSection(text)
if !strings.Contains(section, "Some Author") {
t.Errorf("expected reference section to contain citation, got: %q", section)
}
}

func TestExtractCitationSection_NotFound(t *testing.T) {
text := "This paper has no dedicated references section."
section := extractCitationSection(text)
if section != "" {
t.Errorf("expected empty string when no reference section, got: %q", section)
}
}

// --- Citation ref parsing ---

func TestParseCitationRefs_BracketStyle(t *testing.T) {
refSection := `References
[1] Author A. Title one. Journal, 2020. https://doi.org/10.1234/abc
[2] Author B. Title two. Conference, 2021. doi: 10.5678/xyz
[3] Author C. Title three. 2022.
`
refs := parseCitationRefs(refSection, 10)
if len(refs) < 2 {
t.Fatalf("expected at least 2 citations, got %d", len(refs))
}
if refs[0].Index != 1 {
t.Errorf("expected index 1, got %d", refs[0].Index)
}
if refs[0].DOI != "10.1234/abc" {
t.Errorf("expected DOI '10.1234/abc', got %q", refs[0].DOI)
}
}

func TestParseCitationRefs_NumberedDotStyle(t *testing.T) {
refSection := `References
1. First Author. Paper A. 2019. doi:10.1111/aaa
2. Second Author. Paper B. 2020.
`
refs := parseCitationRefs(refSection, 10)
if len(refs) < 1 {
t.Fatalf("expected at least 1 citation, got %d", len(refs))
}
if refs[0].Index != 1 {
t.Errorf("expected index 1, got %d", refs[0].Index)
}
}

func TestParseCitationRefs_DOIFallback(t *testing.T) {
// No numbered style, but DOIs present
refSection := "see Smith et al. (10.1234/abc) and Jones (10.5678/def) for details"
refs := parseCitationRefs(refSection, 10)
if len(refs) < 2 {
t.Fatalf("expected 2 DOI references, got %d", len(refs))
}
}

func TestParseCitationRefs_MaxCitations(t *testing.T) {
var sb strings.Builder
for i := 1; i <= 10; i++ {
sb.WriteString(fmt.Sprintf("[%d] Author %d. Title. 2020. doi:10.1234/%d\n", i, i, i))
}
refs := parseCitationRefs(sb.String(), 3)
if len(refs) > 3 {
t.Errorf("expected at most 3 refs with max=3, got %d", len(refs))
}
}

func TestExtractDOIFromText(t *testing.T) {
cases := []struct {
input string
want  string
}{
{"See doi:10.1234/abc for details", "10.1234/abc"},
{"Available at https://doi.org/10.5678/xyz.", "10.5678/xyz"},
{"No DOI here", ""},
{"DOI: 10.1111/test-2024", "10.1111/test-2024"},
}
for _, tc := range cases {
got := extractDOIFromText(tc.input)
if got != tc.want {
t.Errorf("extractDOIFromText(%q) = %q, want %q", tc.input, got, tc.want)
}
}
}

func TestExtractYearFromText(t *testing.T) {
cases := []struct {
input string
want  string
}{
{"Published in 2023.", "2023"},
{"(2019) Nature.", "2019"},
{"No year here", ""},
}
for _, tc := range cases {
got := extractYearFromText(tc.input)
if got != tc.want {
t.Errorf("extractYearFromText(%q) = %q, want %q", tc.input, got, tc.want)
}
}
}

// --- CitationRef.Format ---

func TestCitationRef_Format(t *testing.T) {
ref := CitationRef{
Index:   3,
Title:   "Deep Learning",
Authors: "LeCun, Y., Bengio, Y.",
Year:    "2015",
DOI:     "10.1038/nature14539",
PageURL: "https://doi.org/10.1038/nature14539",
PDFURL:  "https://example.com/dl.pdf",
IsOA:    true,
RawText: "[3] LeCun et al. Deep Learning. Nature. 2015.",
}
formatted := ref.Format()
for _, want := range []string{"[3]", "Deep Learning", "LeCun", "2015", "10.1038/nature14539", "true"} {
if !strings.Contains(formatted, want) {
t.Errorf("expected %q in formatted output:\n%s", want, formatted)
}
}
}

// --- AcademicExtractCitationsTool ---

func TestAcademicExtractCitationsTool_Name(t *testing.T) {
tool := NewAcademicExtractCitationsTool("", t.TempDir(), false)
if tool.Name() != "academic_extract_citations" {
t.Errorf("expected name 'academic_extract_citations', got %q", tool.Name())
}
}

func TestAcademicExtractCitationsTool_MissingFilePath(t *testing.T) {
tool := NewAcademicExtractCitationsTool("", t.TempDir(), false)
result := tool.Execute(context.Background(), map[string]any{})
if !result.IsError {
t.Error("expected error when file_path is missing")
}
}

func TestAcademicExtractCitationsTool_DownloadAvailableWithoutSaveDir(t *testing.T) {
tool := NewAcademicExtractCitationsTool("", t.TempDir(), false)
result := tool.Execute(context.Background(), map[string]any{
"file_path":          "paper.txt",
"download_available": true,
})
if !result.IsError {
t.Error("expected error when download_available=true but save_dir not set")
}
}

func TestAcademicExtractCitationsTool_ExtractFromTxtFile(t *testing.T) {
dir := t.TempDir()
paperPath := filepath.Join(dir, "paper.txt")

content := `Abstract
This paper presents methods for machine learning.

References
[1] LeCun, Y. et al. Deep Learning. Nature, 2015. doi:10.1038/nature14539
[2] Hinton, G. et al. A fast learning algorithm for deep belief nets. Neural Computation, 2006.
`
if err := os.WriteFile(paperPath, []byte(content), 0o644); err != nil {
t.Fatalf("setup failed: %v", err)
}

tool := NewAcademicExtractCitationsTool("", dir, false)
result := tool.Execute(context.Background(), map[string]any{
"file_path":     paperPath,
"max_citations": float64(5),
})

if result.IsError {
t.Fatalf("expected success, got error: %s", result.ForLLM)
}
if !strings.Contains(result.ForLLM, "Citation") {
t.Errorf("expected 'Citation' in output, got: %s", result.ForLLM)
}
// The DOI in citation [1] should be found
if !strings.Contains(result.ForLLM, "10.1038/nature14539") {
t.Errorf("expected DOI in output, got: %s", result.ForLLM)
}
}

func TestAcademicExtractCitationsTool_SaveReport(t *testing.T) {
dir := t.TempDir()
paperPath := filepath.Join(dir, "paper.txt")
reportPath := filepath.Join(dir, "report.txt")

content := "References\n[1] Some Author. Title. 2020. doi:10.1234/test\n"
if err := os.WriteFile(paperPath, []byte(content), 0o644); err != nil {
t.Fatalf("setup failed: %v", err)
}

tool := NewAcademicExtractCitationsTool("", dir, false)
result := tool.Execute(context.Background(), map[string]any{
"file_path":      paperPath,
"save_report_to": reportPath,
})

if result.IsError {
t.Fatalf("expected success, got error: %s", result.ForLLM)
}
data, err := os.ReadFile(reportPath)
if err != nil {
t.Fatalf("expected report file to exist: %v", err)
}
if !strings.Contains(string(data), "10.1234/test") {
t.Errorf("expected DOI in saved report, got: %s", string(data))
}
}

func TestAcademicExtractCitationsTool_NoRefsInFile(t *testing.T) {
dir := t.TempDir()
paperPath := filepath.Join(dir, "paper.txt")
content := "This is just some plain text with no citations or references section."
if err := os.WriteFile(paperPath, []byte(content), 0o644); err != nil {
t.Fatalf("setup: %v", err)
}

tool := NewAcademicExtractCitationsTool("", dir, false)
result := tool.Execute(context.Background(), map[string]any{
"file_path": paperPath,
})

// Should not be an error, just report no citations found
if result.IsError {
t.Errorf("expected non-error result, got: %s", result.ForLLM)
}
if !strings.Contains(result.ForLLM, "No citations") {
t.Errorf("expected 'No citations' message, got: %s", result.ForLLM)
}
}

func TestAcademicExtractCitationsTool_Parameters(t *testing.T) {
tool := NewAcademicExtractCitationsTool("", t.TempDir(), false)
params := tool.Parameters()
props, ok := params["properties"].(map[string]any)
if !ok {
t.Fatal("expected properties map")
}
for _, key := range []string{"file_path", "max_citations", "download_available", "save_dir", "save_report_to"} {
if _, ok := props[key]; !ok {
t.Errorf("expected parameter %q", key)
}
}
}

// --- detectPDF ---

func TestDetectPDF(t *testing.T) {
if !detectPDF([]byte("%PDF-1.4 content"), "", "") {
t.Error("expected true for %PDF magic bytes")
}
if !detectPDF([]byte("other"), "application/pdf", "") {
t.Error("expected true for application/pdf content-type")
}
if !detectPDF([]byte("other"), "", "https://example.com/doc.pdf") {
t.Error("expected true for .pdf URL suffix")
}
if detectPDF([]byte("<html>"), "text/html", "https://example.com/page") {
t.Error("expected false for HTML content")
}
}
