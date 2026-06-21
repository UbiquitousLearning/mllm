package tui

import (
	"fmt"
	"os"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/muesli/termenv"
	"golang.org/x/term"
)

// Model stores the application state
type welcomeModel struct {
	input    string          // Content in the input field
	messages []string        // List of displayed messages
	width    int             // Terminal width
	height   int             // Terminal height
	term     *termenv.Output // Terminal environment
}

// Generate the specified ASCII art title "MLLM TEAM"
func asciiTitle() string {
	return `███    ███ ██      ██      ███    ███     ████████ ███████  █████  ███    ███ 
████  ████ ██      ██      ████  ████        ██    ██      ██   ██ ████  ████ 
██ ████ ██ ██      ██      ██ ████ ██        ██    █████   ███████ ██ ████ ██ 
██  ██  ██ ██      ██      ██  ██  ██        ██    ██      ██   ██ ██  ██  ██ 
██      ██ ███████ ███████ ██      ██        ██    ███████ ██   ██ ██      ██`
}

// Get the number of lines in the ASCII title
func asciiTitleLines() int {
	return strings.Count(asciiTitle(), "\n") + 1 // Add 1 because the last line has no newline character
}

// InitialModel returns the initial model
func InitialWelcomeModel() welcomeModel {
	// Get initial terminal dimensions
	w, h := 120, 30 // Default dimensions

	if term.IsTerminal(int(os.Stdout.Fd())) {
		if sw, sh, err := term.GetSize(int(os.Stdout.Fd())); err == nil {
			w, h = sw, sh
		}
	}

	return welcomeModel{
		messages: []string{
			"Welcome to MLLM TEAM Assistant",
			"",
			"Please enter your message or command:",
			"- Type your question to get assistance",
			"- /clear to clear conversation",
			"- /exit to quit",
		},
		width:  w,
		height: h,
		term:   termenv.NewOutput(os.Stdout),
	}
}

// Init initializes the model
func (m welcomeModel) Init() tea.Cmd {
	return nil
}

// Update handles messages and updates the model
func (m welcomeModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
			return m, tea.Quit

		case tea.KeyEnter:
			if m.input != "" {
				if m.input == "/clear" {
					m.messages = []string{"Conversation cleared"}
				} else if m.input == "/exit" {
					return m, tea.Quit
				} else {
					m.messages = append(m.messages, "> "+m.input)
					reply := fmt.Sprintf("MLLM TEAM: We're processing your request: %s", m.input)
					m.messages = append(m.messages, reply)
				}
				m.input = ""
			}
			return m, nil

		case tea.KeyBackspace:
			if len(m.input) > 0 {
				m.input = m.input[:len(m.input)-1]
			}
			return m, nil

		default:
			// Check for inputtable characters
			if msg.String() != "" && len(msg.Runes) > 0 {
				m.input += msg.String()
			}
			return m, nil
		}

	case tea.WindowSizeMsg:
		// 从窗口大小消息更新尺寸
		m.width = msg.Width
		m.height = msg.Height
		return m, nil
	}
	return m, nil
}

// View renders the interface
func (m welcomeModel) View() string {
	var b strings.Builder

	// Draw ASCII title
	title := asciiTitle()
	lines := strings.Split(title, "\n")

	// Center the title
	for _, line := range lines {
		if len(line) < m.width {
			padding := (m.width - len(line)) / 2
			b.WriteString(strings.Repeat(" ", padding) + line + "\n")
		} else {
			// If terminal width is less than title width, display the part that fits
			b.WriteString(line[:m.width] + "\n")
		}
	}

	// Draw separator
	b.WriteString(strings.Repeat("─", m.width) + "\n")

	// Calculate available space for message area
	titleLines := asciiTitleLines()
	messageLines := m.height - titleLines - 6 // Subtract title, separator and input area

	if messageLines < 1 {
		messageLines = 1
	}

	// Display only the latest messages that fit in the space
	startIdx := 0
	if len(m.messages) > messageLines {
		startIdx = len(m.messages) - messageLines
	}

	// Display messages
	for _, msg := range m.messages[startIdx:] {
		wrappedLines := wrapText(msg, m.width)
		for _, line := range wrappedLines {
			b.WriteString(line + "\n")
		}
	}

	// Fill blank lines
	remainingLines := messageLines - (len(m.messages) - startIdx)
	if remainingLines > 0 {
		b.WriteString(strings.Repeat("\n", remainingLines))
	}

	// Draw separator
	b.WriteString(strings.Repeat("─", m.width) + "\n")

	// Draw input box
	inputLine := "> " + m.input
	if len(inputLine) > m.width {
		inputLine = inputLine[len(inputLine)-m.width+1:]
	}
	b.WriteString(inputLine)

	// Fill to terminal width
	if len(inputLine) < m.width {
		b.WriteString(strings.Repeat(" ", m.width-len(inputLine)))
	}

	return b.String()
}

// Text wrapping helper function
func wrapText(text string, width int) []string {
	var lines []string
	currentLine := ""

	for _, word := range strings.Fields(text) {
		if len(currentLine)+len(word)+1 > width {
			lines = append(lines, currentLine)
			currentLine = word
		} else {
			if currentLine == "" {
				currentLine = word
			} else {
				currentLine += " " + word
			}
		}
	}

	if currentLine != "" {
		lines = append(lines, currentLine)
	}

	return lines
}

// RunWelcome runs the welcome chat interface
func RunWelcome() error {
	p := tea.NewProgram(InitialWelcomeModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		return fmt.Errorf("error running welcome interface: %v", err)
	}
	return nil
}
