"""
Synonym Provider Module - Provides word alternatives/synonyms.
Built-in synonym database without external dependencies.
Supports LLM-based synonym generation when model is available.
"""

from typing import List, Optional, Dict, Callable
from dataclasses import dataclass


@dataclass
class SynonymResult:
    """Result of synonym lookup."""
    word: str
    synonyms: List[str]
    source: str  # "database" or "llm"


# Built-in synonym database (common words)
SYNONYM_DATABASE: Dict[str, List[str]] = {
    # Common verbs
    "use": ["utilize", "employ", "apply", "leverage", "work with"],
    "get": ["obtain", "acquire", "receive", "gain", "fetch"],
    "make": ["create", "build", "construct", "produce", "generate"],
    "give": ["provide", "offer", "supply", "deliver", "present"],
    "take": ["grab", "seize", "capture", "accept", "receive"],
    "see": ["observe", "notice", "view", "witness", "spot"],
    "come": ["arrive", "approach", "reach", "appear", "show up"],
    "go": ["leave", "depart", "proceed", "travel", "move"],
    "know": ["understand", "realize", "recognize", "comprehend", "grasp"],
    "think": ["believe", "consider", "suppose", "assume", "reckon"],
    "want": ["desire", "wish", "need", "require", "crave"],
    "look": ["appear", "seem", "glance", "gaze", "observe"],
    "say": ["state", "mention", "declare", "express", "tell"],
    "find": ["discover", "locate", "uncover", "detect", "identify"],
    "put": ["place", "set", "position", "lay", "deposit"],
    "try": ["attempt", "endeavor", "strive", "aim", "seek"],
    "ask": ["inquire", "question", "request", "query", "demand"],
    "work": ["function", "operate", "perform", "labor", "toil"],
    "seem": ["appear", "look", "sound", "feel", "come across as"],
    "feel": ["sense", "experience", "perceive", "notice", "detect"],
    "become": ["turn into", "grow", "develop into", "transform into", "evolve into"],
    "leave": ["depart", "exit", "abandon", "vacate", "quit"],
    "call": ["phone", "contact", "ring", "name", "summon"],
    "keep": ["maintain", "retain", "preserve", "hold", "store"],
    "let": ["allow", "permit", "enable", "authorize", "grant"],
    "begin": ["start", "commence", "initiate", "launch", "kick off"],
    "show": ["display", "demonstrate", "reveal", "exhibit", "present"],
    "hear": ["listen to", "catch", "perceive", "detect", "pick up"],
    "help": ["assist", "aid", "support", "facilitate", "enable"],
    "talk": ["speak", "discuss", "chat", "converse", "communicate"],
    "turn": ["rotate", "spin", "twist", "pivot", "revolve"],
    "start": ["begin", "commence", "initiate", "launch", "originate"],
    "run": ["operate", "manage", "execute", "sprint", "dash"],
    "move": ["shift", "transfer", "relocate", "budge", "transport"],
    "live": ["reside", "dwell", "inhabit", "exist", "survive"],
    "believe": ["think", "trust", "accept", "consider", "suppose"],
    "hold": ["grasp", "grip", "clutch", "carry", "contain"],
    "bring": ["carry", "deliver", "transport", "fetch", "convey"],
    "happen": ["occur", "take place", "transpire", "arise", "unfold"],
    "write": ["compose", "draft", "pen", "author", "create"],
    "provide": ["supply", "offer", "give", "furnish", "deliver"],
    "sit": ["rest", "settle", "perch", "be seated", "remain"],
    "stand": ["rise", "remain upright", "endure", "tolerate", "withstand"],
    "lose": ["misplace", "forfeit", "drop", "miss", "fail"],
    "pay": ["compensate", "reimburse", "settle", "remit", "disburse"],
    "meet": ["encounter", "greet", "gather", "assemble", "convene"],
    "include": ["contain", "comprise", "encompass", "incorporate", "cover"],
    "continue": ["proceed", "persist", "carry on", "resume", "maintain"],
    "set": ["place", "put", "position", "establish", "configure"],
    "learn": ["discover", "study", "master", "acquire", "absorb"],
    "change": ["alter", "modify", "adjust", "transform", "vary"],
    "lead": ["guide", "direct", "head", "conduct", "steer"],
    "understand": ["comprehend", "grasp", "realize", "appreciate", "fathom"],
    "watch": ["observe", "view", "monitor", "witness", "survey"],
    "follow": ["pursue", "trail", "track", "succeed", "obey"],
    "stop": ["cease", "halt", "end", "quit", "discontinue"],
    "create": ["make", "produce", "generate", "develop", "design"],
    "speak": ["talk", "say", "express", "articulate", "communicate"],
    "read": ["peruse", "study", "scan", "review", "examine"],
    "allow": ["permit", "let", "enable", "authorize", "sanction"],
    "add": ["include", "append", "attach", "incorporate", "insert"],
    "grow": ["expand", "increase", "develop", "flourish", "thrive"],
    "open": ["unlock", "unfasten", "uncover", "reveal", "expose"],
    "walk": ["stroll", "stride", "march", "wander", "amble"],
    "win": ["triumph", "succeed", "prevail", "conquer", "achieve"],
    "offer": ["provide", "present", "propose", "suggest", "extend"],
    "remember": ["recall", "recollect", "retain", "reminisce", "think back"],
    "love": ["adore", "cherish", "treasure", "appreciate", "admire"],
    "consider": ["contemplate", "ponder", "reflect on", "think about", "weigh"],
    "appear": ["seem", "look", "emerge", "surface", "show up"],
    "buy": ["purchase", "acquire", "obtain", "get", "procure"],
    "wait": ["await", "hold on", "stay", "remain", "pause"],
    "serve": ["assist", "help", "aid", "attend", "cater to"],
    "die": ["perish", "expire", "pass away", "succumb", "decease"],
    "send": ["dispatch", "transmit", "forward", "deliver", "ship"],
    "expect": ["anticipate", "await", "predict", "foresee", "count on"],
    "build": ["construct", "erect", "create", "develop", "assemble"],
    "stay": ["remain", "linger", "wait", "dwell", "reside"],
    "fall": ["drop", "tumble", "plunge", "descend", "collapse"],
    "cut": ["slice", "trim", "chop", "sever", "reduce"],
    "reach": ["arrive at", "attain", "achieve", "extend to", "contact"],
    "kill": ["slay", "eliminate", "destroy", "end", "terminate"],
    "remain": ["stay", "continue", "persist", "endure", "last"],

    # Common adjectives
    "good": ["great", "excellent", "fine", "nice", "wonderful"],
    "bad": ["poor", "terrible", "awful", "dreadful", "horrible"],
    "big": ["large", "huge", "massive", "enormous", "giant"],
    "small": ["little", "tiny", "mini", "compact", "petite"],
    "new": ["fresh", "recent", "modern", "novel", "latest"],
    "old": ["ancient", "aged", "elderly", "vintage", "antique"],
    "important": ["significant", "crucial", "vital", "essential", "key"],
    "different": ["distinct", "various", "diverse", "unique", "varied"],
    "hard": ["difficult", "tough", "challenging", "demanding", "arduous"],
    "easy": ["simple", "effortless", "straightforward", "uncomplicated", "painless"],
    "fast": ["quick", "rapid", "swift", "speedy", "hasty"],
    "slow": ["gradual", "unhurried", "leisurely", "sluggish", "plodding"],
    "happy": ["joyful", "cheerful", "delighted", "pleased", "content"],
    "sad": ["unhappy", "sorrowful", "dejected", "melancholy", "gloomy"],
    "beautiful": ["gorgeous", "lovely", "stunning", "attractive", "pretty"],
    "ugly": ["unattractive", "hideous", "unsightly", "grotesque", "homely"],
    "strong": ["powerful", "robust", "sturdy", "tough", "mighty"],
    "weak": ["feeble", "frail", "fragile", "delicate", "powerless"],
    "hot": ["warm", "heated", "boiling", "scorching", "burning"],
    "cold": ["cool", "chilly", "freezing", "icy", "frigid"],
    "right": ["correct", "accurate", "proper", "appropriate", "suitable"],
    "wrong": ["incorrect", "inaccurate", "mistaken", "erroneous", "false"],
    "long": ["lengthy", "extended", "prolonged", "extensive", "enduring"],
    "short": ["brief", "concise", "compact", "abbreviated", "limited"],
    "high": ["tall", "elevated", "lofty", "towering", "raised"],
    "low": ["short", "shallow", "reduced", "minimal", "diminished"],
    "clear": ["obvious", "evident", "apparent", "transparent", "plain"],
    "dark": ["dim", "shadowy", "murky", "gloomy", "obscure"],
    "light": ["bright", "luminous", "radiant", "glowing", "illuminated"],
    "full": ["complete", "entire", "whole", "packed", "loaded"],
    "empty": ["vacant", "void", "hollow", "bare", "blank"],
    "rich": ["wealthy", "affluent", "prosperous", "well-off", "loaded"],
    "poor": ["impoverished", "needy", "destitute", "broke", "penniless"],
    "young": ["youthful", "juvenile", "adolescent", "immature", "fresh"],
    "real": ["genuine", "authentic", "true", "actual", "legitimate"],
    "fake": ["false", "counterfeit", "artificial", "bogus", "phony"],
    "same": ["identical", "alike", "similar", "equivalent", "matching"],

    # Common adverbs
    "very": ["extremely", "highly", "really", "truly", "exceptionally"],
    "quickly": ["rapidly", "swiftly", "speedily", "hastily", "promptly"],
    "slowly": ["gradually", "leisurely", "unhurriedly", "steadily", "carefully"],
    "often": ["frequently", "regularly", "commonly", "repeatedly", "usually"],
    "always": ["constantly", "perpetually", "forever", "continuously", "invariably"],
    "never": ["not ever", "at no time", "not once", "under no circumstances"],
    "sometimes": ["occasionally", "periodically", "now and then", "at times", "once in a while"],
    "really": ["truly", "genuinely", "actually", "honestly", "indeed"],
    "almost": ["nearly", "practically", "virtually", "about", "approximately"],
    "probably": ["likely", "possibly", "perhaps", "presumably", "maybe"],
    "actually": ["really", "truly", "in fact", "indeed", "genuinely"],
    "also": ["too", "as well", "additionally", "besides", "moreover"],
    "just": ["only", "merely", "simply", "exactly", "precisely"],
    "still": ["yet", "even now", "nevertheless", "nonetheless", "however"],
    "already": ["previously", "before now", "by now", "so soon", "earlier"],
    "enough": ["sufficiently", "adequately", "fairly", "reasonably", "quite"],
    "especially": ["particularly", "specifically", "notably", "mainly", "chiefly"],
    "certainly": ["definitely", "surely", "absolutely", "undoubtedly", "positively"],

    # Common nouns
    "thing": ["item", "object", "matter", "element", "aspect"],
    "person": ["individual", "human", "someone", "somebody", "character"],
    "people": ["individuals", "folks", "persons", "humans", "citizens"],
    "place": ["location", "spot", "area", "site", "venue"],
    "time": ["period", "moment", "occasion", "instance", "duration"],
    "way": ["method", "manner", "approach", "means", "mode"],
    "problem": ["issue", "challenge", "difficulty", "trouble", "obstacle"],
    "idea": ["concept", "thought", "notion", "plan", "suggestion"],
    "part": ["portion", "section", "piece", "segment", "component"],
    "result": ["outcome", "consequence", "effect", "product", "finding"],
    "reason": ["cause", "motive", "purpose", "grounds", "basis"],
    "fact": ["truth", "reality", "detail", "point", "datum"],
    "change": ["shift", "alteration", "modification", "transformation", "variation"],
    "example": ["instance", "case", "illustration", "sample", "specimen"],
    "group": ["team", "collection", "set", "bunch", "cluster"],
    "world": ["globe", "planet", "earth", "universe", "realm"],
    "end": ["conclusion", "finish", "close", "termination", "finale"],
    "kind": ["type", "sort", "variety", "category", "class"],
    "answer": ["response", "reply", "solution", "reaction", "feedback"],
    "question": ["query", "inquiry", "issue", "matter", "problem"],
}


class SynonymProvider:
    """Provides synonyms for words using built-in database and optional LLM."""

    def __init__(self):
        self.database = SYNONYM_DATABASE
        self.model = None
        self.model_available = False

    def set_model(self, model):
        """Set LLM model for advanced synonym generation."""
        self.model = model
        self.model_available = model is not None

    def get_synonyms(self, word: str, use_llm: bool = False,
                     max_synonyms: int = 5) -> SynonymResult:
        """
        Get synonyms for a word.

        Args:
            word: Word to find synonyms for
            use_llm: Whether to use LLM for better synonyms
            max_synonyms: Maximum number of synonyms to return

        Returns:
            SynonymResult with list of synonyms
        """
        word_lower = word.lower().strip()

        # First check database
        if word_lower in self.database:
            synonyms = self.database[word_lower][:max_synonyms]
            return SynonymResult(
                word=word,
                synonyms=synonyms,
                source="database"
            )

        # Try LLM if available and requested
        if use_llm and self.model_available:
            llm_synonyms = self._get_llm_synonyms(word, max_synonyms)
            if llm_synonyms:
                return SynonymResult(
                    word=word,
                    synonyms=llm_synonyms,
                    source="llm"
                )

        # No synonyms found
        return SynonymResult(
            word=word,
            synonyms=[],
            source="none"
        )

    def get_synonyms_batch(self, words: List[str], use_llm: bool = False,
                           max_synonyms: int = 5) -> Dict[str, SynonymResult]:
        """Get synonyms for multiple words."""
        results = {}
        for word in words:
            results[word] = self.get_synonyms(word, use_llm, max_synonyms)
        return results

    def has_synonyms(self, word: str) -> bool:
        """Check if word has synonyms in database."""
        return word.lower().strip() in self.database

    def add_custom_synonyms(self, word: str, synonyms: List[str]) -> None:
        """Add custom synonyms for a word."""
        word_lower = word.lower().strip()
        if word_lower in self.database:
            # Merge with existing, avoiding duplicates
            existing = set(self.database[word_lower])
            for syn in synonyms:
                if syn.lower() not in existing:
                    self.database[word_lower].append(syn)
        else:
            self.database[word_lower] = synonyms

    def _get_llm_synonyms(self, word: str, max_synonyms: int) -> List[str]:
        """Get synonyms using LLM."""
        if not self.model:
            return []

        prompt = f"""<|im_start|>system
You are a synonym expert. Provide exactly {max_synonyms} synonyms for the given word.
Return ONLY the synonyms, one per line, no numbering or explanations.<|im_end|>
<|im_start|>user
Word: {word}<|im_end|>
<|im_start|>assistant
"""

        try:
            response = self.model(
                prompt,
                max_tokens=50,
                temperature=0.3,
                top_p=0.9,
                stop=["<|im_end|>", "<|im_start|>"]
            )

            result = response["choices"][0]["text"].strip()
            synonyms = [s.strip() for s in result.split('\n') if s.strip()]
            return synonyms[:max_synonyms]

        except Exception:
            return []

    def get_word_count(self) -> int:
        """Get number of words in database."""
        return len(self.database)


# Singleton instance
_synonym_provider = SynonymProvider()


def get_synonyms(word: str, use_llm: bool = False) -> List[str]:
    """Convenience function to get synonyms."""
    result = _synonym_provider.get_synonyms(word, use_llm)
    return result.synonyms


def set_llm_model(model) -> None:
    """Set LLM model for synonym provider."""
    _synonym_provider.set_model(model)


def get_synonym_provider() -> SynonymProvider:
    """Get the global synonym provider instance."""
    return _synonym_provider
