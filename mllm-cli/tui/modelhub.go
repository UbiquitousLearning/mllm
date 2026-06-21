// Copyright (c) MLLM Team.
// Licensed under the MIT License.

package tui

import (
	"fmt"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	titleStyle        = lipgloss.NewStyle().MarginLeft(2)
	itemStyle         = lipgloss.NewStyle().PaddingLeft(4)
	selectedItemStyle = lipgloss.NewStyle().PaddingLeft(2).Foreground(lipgloss.Color("170"))
	paginationStyle   = lipgloss.NewStyle().PaddingLeft(4).Foreground(lipgloss.Color("240"))
	helpStyle         = lipgloss.NewStyle().PaddingTop(1).PaddingLeft(4).Foreground(lipgloss.Color("240"))
	quitStyle         = lipgloss.NewStyle().Margin(1, 0, 2, 4)
	urlStyle          = lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Italic(true)
	errorStyle        = lipgloss.NewStyle().Foreground(lipgloss.Color("196")).Bold(true)
)

type modelItem struct {
	Name       string `json:"name"`
	URL        string `json:"url"`
	Size       string `json:"size"`
	Quantizer  string `json:"quantizer"`
	Format     string `json:"format"`
	Downloaded bool   `json:"downloaded"`
}

func (m modelItem) Title() string { return m.Name }
func (m modelItem) Description() string {
	return fmt.Sprintf("Size: %s | Quantizer: %s | Format: %s", m.Size, m.Quantizer, m.Format)
}
func (m modelItem) FilterValue() string { return m.Name }

type modelHubModel struct {
	list         list.Model
	models       []modelItem
	selectedURL  string
	loading      bool
	spinner      spinner.Model
	err          error
	quitting     bool
	downloadProg float64
}

func (m modelHubModel) Init() tea.Cmd {
	return tea.Batch(m.spinner.Tick, fetchModels)
}

func (m modelHubModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			m.quitting = true
			return m, tea.Quit
		case "enter":
			if item, ok := m.list.SelectedItem().(modelItem); ok {
				m.selectedURL = item.URL
				return m, tea.Quit
			}
		}

	case modelsMsg:
		m.models = msg
		items := make([]list.Item, len(msg))
		for i, model := range msg {
			items[i] = model
		}
		m.list.SetItems(items)
		m.loading = false

	case errMsg:
		m.err = msg
		m.loading = false

	case spinner.TickMsg:
		if m.loading {
			m.spinner, cmd = m.spinner.Update(msg)
			cmds = append(cmds, cmd)
		}
	}

	m.list, cmd = m.list.Update(msg)
	cmds = append(cmds, cmd)

	return m, tea.Batch(cmds...)
}

func (m modelHubModel) View() string {
	if m.quitting {
		return quitStyle.Render("Thanks for using MLLM ModelHub!")
	}

	if m.err != nil {
		return errorStyle.Render(fmt.Sprintf("Error: %v", m.err))
	}

	if m.loading {
		return fmt.Sprintf("\n\n   %s Loading models from hub...\n\n", m.spinner.View())
	}

	if m.selectedURL != "" {
		item := m.list.SelectedItem().(modelItem)
		s := fmt.Sprintf("\n\n   Selected model: %s\n", item.Name)
		s += fmt.Sprintf("   Size: %s | Quantizer: %s | Format: %s\n", item.Size, item.Quantizer, item.Format)
		s += fmt.Sprintf("   Downloading from: %s\n\n", m.selectedURL)
		if m.downloadProg > 0 {
			s += fmt.Sprintf("   Download progress: %.1f%%\n\n", m.downloadProg*100)
		}
		s += "   Press 'q' to quit\n"
		return s
	}

	s := "\n" + titleStyle.Render("MLLM ModelHub - Select a model to download") + "\n\n"
	s += m.list.View()
	s += helpStyle.Render("\n↑/↓: navigate • enter: select model • q: quit\n")
	return s
}

type modelsMsg []modelItem
type errMsg error

func fetchModels() tea.Msg {
	// This would normally fetch from a real URL
	// For now, we'll return sample data
	// In a real implementation, you would do:
	/*
		resp, err := http.Get("https://your-model-hub-url.com/models.json")
		if err != nil {
			return errMsg(err)
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return errMsg(err)
		}

		var models []modelItem
		err = json.Unmarshal(body, &models)
		if err != nil {
			return errMsg(err)
		}

		return modelsMsg(models)
	*/

	// Sample data for demonstration
	models := []modelItem{
		{
			Name:      "Qwen2-VL-7B-Instruct",
			URL:       "https://model-hub.com/models/qwen2-vl-7b-instruct.gguf",
			Size:      "4.2 GB",
			Quantizer: "GGUF Q4_K_M",
			Format:    "GGUF",
		},
		{
			Name:      "Phi-3-mini-4k-instruct",
			URL:       "https://model-hub.com/models/phi-3-mini-4k-instruct.gguf",
			Size:      "2.1 GB",
			Quantizer: "GGUF Q4_K_M",
			Format:    "GGUF",
		},
		{
			Name:      "Llama-3.1-8B-Instruct",
			URL:       "https://model-hub.com/models/llama-3-1-8b-instruct.gguf",
			Size:      "4.9 GB",
			Quantizer: "GGUF Q4_K_M",
			Format:    "GGUF",
		},
		{
			Name:      "Gemma-2-9B-it",
			URL:       "https://model-hub.com/models/gemma-2-9b-it.gguf",
			Size:      "5.2 GB",
			Quantizer: "GGUF Q4_K_M",
			Format:    "GGUF",
		},
		{
			Name:      "Mistral-7B-Instruct-v0.3",
			URL:       "https://model-hub.com/models/mistral-7b-instruct-v0.3.gguf",
			Size:      "4.1 GB",
			Quantizer: "GGUF Q4_K_M",
			Format:    "GGUF",
		},
	}

	return modelsMsg(models)
}

func downloadModel(url string) tea.Msg {
	// This would handle the actual model download
	// For now, just return the URL
	return url
}

func NewModelHub() (*modelHubModel, error) {
	// Create a list of items
	items := []list.Item{}

	// Create the list model
	l := list.New(items, list.NewDefaultDelegate(), 0, 0)
	l.Title = "MLLM Models"
	l.Styles.Title = titleStyle
	l.Styles.PaginationStyle = paginationStyle

	// Create the spinner
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))

	m := &modelHubModel{
		list:    l,
		loading: true,
		spinner: s,
	}

	return m, nil
}

func RunModelHub() (string, error) {
	model, err := NewModelHub()
	if err != nil {
		return "", err
	}

	p := tea.NewProgram(*model)
	result, err := p.Run()
	if err != nil {
		return "", err
	}

	finalModel := result.(modelHubModel)
	return finalModel.selectedURL, nil
}
