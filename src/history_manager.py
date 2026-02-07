"""
History Manager Module - Saves and restores humanization sessions.
Stores the last N sessions with original/humanized text pairs.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class HistoryEntry:
    """A single history entry representing one humanization session."""
    id: str
    timestamp: str
    original_text: str
    humanized_text: str
    mode: str
    creativity_level: int
    ai_score_before: Optional[float] = None
    ai_score_after: Optional[float] = None
    word_count_original: int = 0
    word_count_humanized: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoryEntry':
        """Create from dictionary."""
        return cls(**data)

    def get_preview(self, max_chars: int = 100) -> str:
        """Get a preview of the original text."""
        text = self.original_text[:max_chars]
        if len(self.original_text) > max_chars:
            text += "..."
        return text

    def get_formatted_timestamp(self) -> str:
        """Get human-readable timestamp."""
        try:
            dt = datetime.fromisoformat(self.timestamp)
            now = datetime.now()

            if dt.date() == now.date():
                return f"Today at {dt.strftime('%H:%M')}"
            elif (now - dt).days == 1:
                return f"Yesterday at {dt.strftime('%H:%M')}"
            elif (now - dt).days < 7:
                return dt.strftime('%A at %H:%M')
            else:
                return dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            return self.timestamp


class HistoryManager:
    """Manages history of humanization sessions."""

    DEFAULT_CONFIG_DIR = ".kaibot"
    HISTORY_FILE = "history.json"
    MAX_ENTRIES = 50  # Maximum history entries to keep

    def __init__(self, config_dir: Optional[str] = None, max_entries: int = 50):
        """
        Initialize history manager.

        Args:
            config_dir: Custom config directory path
            max_entries: Maximum number of entries to keep
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path.home() / self.DEFAULT_CONFIG_DIR

        self.history_file = self.config_dir / self.HISTORY_FILE
        self.max_entries = max_entries
        self.entries: List[HistoryEntry] = []

        self._ensure_config_dir()
        self.load()

    def _ensure_config_dir(self):
        """Create config directory if it doesn't exist."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create config directory: {e}")

    def _generate_id(self) -> str:
        """Generate unique ID for history entry."""
        return datetime.now().strftime('%Y%m%d%H%M%S%f')

    def add_entry(self, original: str, humanized: str, mode: str,
                  creativity_level: int, ai_score_before: Optional[float] = None,
                  ai_score_after: Optional[float] = None) -> HistoryEntry:
        """
        Add a new history entry.

        Args:
            original: Original text
            humanized: Humanized text
            mode: Paraphrasing mode used
            creativity_level: Creativity slider value
            ai_score_before: AI detection score before humanizing
            ai_score_after: AI detection score after humanizing

        Returns:
            The created HistoryEntry
        """
        entry = HistoryEntry(
            id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            original_text=original,
            humanized_text=humanized,
            mode=mode,
            creativity_level=creativity_level,
            ai_score_before=ai_score_before,
            ai_score_after=ai_score_after,
            word_count_original=len(original.split()),
            word_count_humanized=len(humanized.split())
        )

        # Add to beginning of list
        self.entries.insert(0, entry)

        # Trim to max entries
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[:self.max_entries]

        # Auto-save
        self.save()

        return entry

    def get_entry(self, entry_id: str) -> Optional[HistoryEntry]:
        """Get entry by ID."""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def get_entries(self, limit: int = 10) -> List[HistoryEntry]:
        """Get recent entries."""
        return self.entries[:limit]

    def get_all_entries(self) -> List[HistoryEntry]:
        """Get all entries."""
        return self.entries.copy()

    def delete_entry(self, entry_id: str) -> bool:
        """Delete entry by ID."""
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                del self.entries[i]
                self.save()
                return True
        return False

    def clear_history(self) -> None:
        """Clear all history entries."""
        self.entries = []
        self.save()

    def search_entries(self, query: str) -> List[HistoryEntry]:
        """Search entries by text content."""
        query_lower = query.lower()
        results = []
        for entry in self.entries:
            if (query_lower in entry.original_text.lower() or
                query_lower in entry.humanized_text.lower()):
                results.append(entry)
        return results

    def get_entries_by_mode(self, mode: str) -> List[HistoryEntry]:
        """Get entries filtered by mode."""
        return [e for e in self.entries if e.mode == mode]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about history."""
        if not self.entries:
            return {
                "total_entries": 0,
                "total_words_processed": 0,
                "avg_improvement": 0,
                "most_used_mode": None
            }

        total_words = sum(e.word_count_original for e in self.entries)

        # Calculate average AI score improvement
        improvements = []
        for e in self.entries:
            if e.ai_score_before is not None and e.ai_score_after is not None:
                improvements.append(e.ai_score_before - e.ai_score_after)
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        # Most used mode
        mode_counts: Dict[str, int] = {}
        for e in self.entries:
            mode_counts[e.mode] = mode_counts.get(e.mode, 0) + 1
        most_used = max(mode_counts, key=mode_counts.get) if mode_counts else None

        return {
            "total_entries": len(self.entries),
            "total_words_processed": total_words,
            "avg_improvement": round(avg_improvement, 1),
            "most_used_mode": most_used,
            "mode_breakdown": mode_counts
        }

    def load(self) -> None:
        """Load history from file."""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.entries = [HistoryEntry.from_dict(e) for e in data.get('entries', [])]

        except Exception as e:
            print(f"Warning: Could not load history: {e}")
            self.entries = []

    def save(self) -> bool:
        """Save history to file."""
        try:
            self._ensure_config_dir()

            data = {
                'version': '1.0',
                'entries': [e.to_dict() for e in self.entries]
            }

            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
            return False

    def export_history(self, file_path: str, format_type: str = 'json') -> bool:
        """Export history to file."""
        try:
            if format_type == 'json':
                data = {
                    'exported_at': datetime.now().isoformat(),
                    'entries': [e.to_dict() for e in self.entries]
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            elif format_type == 'txt':
                with open(file_path, 'w', encoding='utf-8') as f:
                    for entry in self.entries:
                        f.write(f"=== {entry.get_formatted_timestamp()} ===\n")
                        f.write(f"Mode: {entry.mode} | Creativity: {entry.creativity_level}%\n\n")
                        f.write("ORIGINAL:\n")
                        f.write(entry.original_text + "\n\n")
                        f.write("HUMANIZED:\n")
                        f.write(entry.humanized_text + "\n\n")
                        f.write("-" * 50 + "\n\n")

            return True
        except Exception:
            return False


# Global instance
_history_manager: Optional[HistoryManager] = None


def get_history_manager() -> HistoryManager:
    """Get the global history manager instance."""
    global _history_manager
    if _history_manager is None:
        _history_manager = HistoryManager()
    return _history_manager


def add_to_history(original: str, humanized: str, mode: str,
                   creativity_level: int, ai_score_before: Optional[float] = None,
                   ai_score_after: Optional[float] = None) -> HistoryEntry:
    """Convenience function to add history entry."""
    return get_history_manager().add_entry(
        original, humanized, mode, creativity_level,
        ai_score_before, ai_score_after
    )


def get_recent_history(limit: int = 10) -> List[HistoryEntry]:
    """Convenience function to get recent history."""
    return get_history_manager().get_entries(limit)
