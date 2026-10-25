//! Test helpers: golden-file comparison.

use std::env;
use std::fs;

/// Compare `actual` against the golden file at `path`.
///
/// The golden is normalized before comparison: lines whose trimmed text
/// starts with `//` are dropped, trailing `//` comments are cut from the
/// remaining lines, trailing whitespace is trimmed per line, and
/// leading/trailing blank lines are dropped. The same normalization is
/// applied to `actual` (which the printer emits comment-free).
///
/// On mismatch a readable diff is printed and the test panics.
/// With `BLESS=1` in the environment the golden file is rewritten with
/// `actual` instead.
pub fn expect_golden(path: &str, actual: &str) {
    if env::var("BLESS").ok().as_deref() == Some("1") {
        fs::write(path, actual).unwrap_or_else(|e| panic!("patina: bless {}: {}", path, e));
        eprintln!("patina: blessed {}", path);
        return;
    }
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("patina: cannot read golden {}: {}", path, e));
    let expected = normalize(&raw);
    let got = normalize(actual);
    if expected != got {
        panic!(
            "patina: golden mismatch for {}\n{}",
            path,
            diff_lines(&expected, &got)
        );
    }
}

fn normalize(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in s.lines() {
        let t = line.trim_end();
        if t.trim_start().starts_with("//") {
            continue;
        }
        let t = match t.find("//") {
            Some(i) => t[..i].trim_end(),
            None => t,
        };
        out.push(t.to_string());
    }
    while out.first().map_or(false, |l| l.is_empty()) {
        out.remove(0);
    }
    while out.last().map_or(false, |l| l.is_empty()) {
        out.pop();
    }
    out
}

/// Minimal LCS line diff: `-` expected (golden), `+` actual (traced).
fn diff_lines(expected: &[String], actual: &[String]) -> String {
    let n = expected.len();
    let m = actual.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if expected[i] == actual[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let mut out = String::from("--- expected (golden)\n+++ actual (traced)\n");
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if expected[i] == actual[j] {
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            out.push_str(&format!("- {}\n", expected[i]));
            i += 1;
        } else {
            out.push_str(&format!("+ {}\n", actual[j]));
            j += 1;
        }
    }
    while i < n {
        out.push_str(&format!("- {}\n", expected[i]));
        i += 1;
    }
    while j < m {
        out.push_str(&format!("+ {}\n", actual[j]));
        j += 1;
    }
    out
}
