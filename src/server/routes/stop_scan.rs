//! Stop strings matched on the decoded text of a content stream.

/// Scans the content stream for the request's stop strings, holding back a
/// tail that could begin one until it is known either way.
///
/// Matching on text rather than on token ids finds a stop string however the
/// model tokenizes it: `four` after a space is one token, after a line break
/// another, inside `fourteen` a third, and all three read as `four`.
pub struct StopScanner {
    stops: Vec<String>,
    held: String,
}

/// What the scanner hands out of a delta of decoded text.
pub enum Scan {
    /// Text safe to hand out, the stream going on.
    Text(String),
    /// The text before a stop string, where the stream ends.
    Stopped(String),
}

impl StopScanner {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            held: String::new(),
        }
    }

    /// Takes the next delta of decoded text.
    pub fn push(&mut self, delta: &str) -> Scan {
        if self.stops.is_empty() {
            return Scan::Text(delta.to_string());
        }
        self.held.push_str(delta);
        if let Some(at) = self.earliest_match() {
            return Scan::Stopped(self.cut(at));
        }
        let keep = self
            .stops
            .iter()
            .map(|stop| longest_suffix_opening(&self.held, stop))
            .max()
            .unwrap_or(0);
        Scan::Text(self.held.drain(..self.held.len() - keep).collect())
    }

    /// Takes the last delta of a stream that ended on its own, releasing
    /// what was held.
    pub fn finish(&mut self, delta: &str) -> Scan {
        if self.stops.is_empty() {
            return Scan::Text(delta.to_string());
        }
        self.held.push_str(delta);
        match self.earliest_match() {
            Some(at) => Scan::Stopped(self.cut(at)),
            None => Scan::Text(std::mem::take(&mut self.held)),
        }
    }

    fn earliest_match(&self) -> Option<usize> {
        self.stops
            .iter()
            .filter_map(|stop| self.held.find(stop.as_str()))
            .min()
    }

    fn cut(&mut self, at: usize) -> String {
        let text = self.held[..at].to_string();
        self.held.clear();
        text
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of `stop`.
fn longest_suffix_opening(text: &str, stop: &str) -> usize {
    (1..stop.len())
        .rev()
        .filter(|&n| stop.is_char_boundary(n))
        .find(|&n| text.ends_with(&stop[..n]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(scan: Scan) -> (String, bool) {
        match scan {
            Scan::Text(t) => (t, false),
            Scan::Stopped(t) => (t, true),
        }
    }

    /// Contract: a stop string split across deltas is found, the text before
    /// it is handed out and nothing from it on.
    #[test]
    fn a_stop_string_split_across_deltas_cuts_the_stream_before_it() {
        let mut scanner = StopScanner::new(vec!["four".to_string()]);
        assert_eq!(
            text(scanner.push("one two th")),
            ("one two th".to_string(), false)
        );
        assert_eq!(text(scanner.push("ree fo")), ("ree ".to_string(), false));
        assert_eq!(text(scanner.push("ur five")), ("".to_string(), true));
    }

    /// Contract: a tail that only looked like the start of a stop string is
    /// released once the stream ends on its own, or once the next delta
    /// shows it was not one.
    #[test]
    fn a_tail_that_was_not_a_stop_is_released() {
        let mut scanner = StopScanner::new(vec!["four".to_string()]);
        assert_eq!(text(scanner.push("fo")), ("".to_string(), false));
        assert_eq!(text(scanner.push("x")), ("fox".to_string(), false));
        assert_eq!(text(scanner.push(" fo")), (" ".to_string(), false));
        assert_eq!(text(scanner.finish("")), ("fo".to_string(), false));

        let mut scanner = StopScanner::new(vec!["four".to_string()]);
        assert_eq!(
            text(scanner.push("three fou")),
            ("three ".to_string(), false)
        );
        assert_eq!(text(scanner.finish("r")), ("".to_string(), true));
    }

    /// Contract: with several stop strings the one found earliest in the
    /// text wins, whatever its length and order.
    #[test]
    fn the_earliest_of_several_stops_wins() {
        let mut scanner = StopScanner::new(vec!["five".to_string(), "two".to_string()]);
        assert_eq!(
            text(scanner.push("one two three five")),
            ("one ".to_string(), true)
        );
    }

    /// Contract: multi-byte text is cut on character boundaries, a stop
    /// string of several characters held back one character at a time.
    #[test]
    fn multibyte_text_is_cut_on_character_boundaries() {
        let mut scanner = StopScanner::new(vec!["日本語".to_string()]);
        assert_eq!(
            text(scanner.push("caffè 日")),
            ("caffè ".to_string(), false)
        );
        assert_eq!(text(scanner.push("本")), ("".to_string(), false));
        assert_eq!(text(scanner.push("語です")), ("".to_string(), true));

        let mut scanner = StopScanner::new(vec!["è".to_string()]);
        assert_eq!(text(scanner.push("caff")), ("caff".to_string(), false));
        assert_eq!(text(scanner.push("è!")), ("".to_string(), true));
    }

    /// Contract: without stop strings every delta passes through untouched.
    #[test]
    fn no_stops_pass_everything_through() {
        let mut scanner = StopScanner::new(Vec::new());
        assert_eq!(text(scanner.push("four")), ("four".to_string(), false));
        assert_eq!(text(scanner.finish(" five")), (" five".to_string(), false));
    }
}
