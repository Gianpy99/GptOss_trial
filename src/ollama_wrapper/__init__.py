from .wrapper import (
	OllamaWrapper,
	OllamaCLIHelper,
	MemoryManager,
	ModelParameters,
	create_coding_assistant,
	create_creative_assistant,
	interactive_repl,
)

# Optional fine-tuning support
try:
	from .finetuning import (
		FineTuningManager,
		create_finetuned_assistant,
	)
	_FINETUNING_AVAILABLE = True
except ImportError:
	_FINETUNING_AVAILABLE = False
	FineTuningManager = None
	create_finetuned_assistant = None

__all__ = [
	"OllamaWrapper",
	"OllamaCLIHelper",
	"MemoryManager",
	"ModelParameters",
	"create_coding_assistant",
	"create_creative_assistant",
	"interactive_repl",
]

# Add fine-tuning exports if available
if _FINETUNING_AVAILABLE:
	__all__.extend([
		"FineTuningManager",
		"create_finetuned_assistant",
	])
