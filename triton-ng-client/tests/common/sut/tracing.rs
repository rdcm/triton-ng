use crate::common::sut::Sut;
use anyhow::{Context, Result};

impl Sut {
    /// Whether the trace of `request_id` places `activity` between its
    /// `REQUEST_START` and its `REQUEST_END`.
    ///
    /// Fails when a trace file that holds the activity is not JSON: the collector
    /// writes the name into the file without escaping it, so a name it cannot
    /// write breaks the file rather than the entry.
    pub(crate) fn activity_is_inside_its_request(
        &self,
        request_id: &str,
        activity: &str,
    ) -> Result<bool> {
        let config = &self.config;

        let Ok(dir) = std::fs::read_dir(config.trace_dir()) else {
            return Ok(false);
        };
        for file in dir {
            let path = file?.path();
            if !path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("trace.json"))
            {
                continue;
            }
            let content = std::fs::read_to_string(&path)?;
            if !content.contains(activity) {
                continue;
            }
            let entries: Vec<serde_json::Value> = serde_json::from_str(&content)
                .with_context(|| format!("{} does not hold JSON", path.display()))?;
            let Some(trace_id) = entries
                .iter()
                .find(|entry| entry["request_id"] == request_id)
                .and_then(|entry| entry["id"].as_u64())
            else {
                continue;
            };

            let mut start = None;
            let mut end = None;
            let mut reported = None;
            for entry in entries
                .iter()
                .filter(|entry| entry["id"].as_u64() == Some(trace_id))
            {
                let timestamps = entry["timestamps"].as_array().into_iter().flatten();
                for timestamp in timestamps {
                    let Some(name) = timestamp["name"].as_str() else {
                        continue;
                    };
                    let Some(ns) = timestamp["ns"].as_u64() else {
                        continue;
                    };
                    if name == activity {
                        reported = Some(ns);
                    } else if name == "REQUEST_START" {
                        start = Some(ns);
                    } else if name == "REQUEST_END" {
                        end = Some(ns);
                    }
                }
            }
            if let (Some(start), Some(reported), Some(end)) = (start, reported, end) {
                if start > reported || reported > end {
                    return Err(anyhow::anyhow!(
                        "the activity {activity} is at {reported}, outside {start}..{end} of request {request_id}"
                    ));
                }
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) async fn set_tracing(&self, level: &str) -> Result<()> {
        let config = &self.config;

        let settings = [
            ("trace_level", level),
            ("trace_rate", "1"),
            ("log_frequency", "1"),
        ];
        self.trace_settings_at(
            settings
                .into_iter()
                .map(|(name, value)| (name.to_string(), vec![value.to_string()]))
                .collect(),
            Some(&config.backend_model),
        )
        .await?;
        Ok(())
    }

    /// Every trace entry written into the trace files so far.
    pub(crate) fn trace_entries(&self) -> Result<Vec<serde_json::Value>> {
        let config = &self.config;

        let mut entries = Vec::new();
        let Ok(dir) = std::fs::read_dir(config.trace_dir()) else {
            return Ok(entries);
        };
        for file in dir {
            let path = file?.path();
            if !path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("trace.json"))
            {
                continue;
            }
            let content = std::fs::read_to_string(&path)?;
            if let Ok(serde_json::Value::Array(values)) = serde_json::from_str(&content) {
                entries.extend(values);
            }
        }
        Ok(entries)
    }
}
