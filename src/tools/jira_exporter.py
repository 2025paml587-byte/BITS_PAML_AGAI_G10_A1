import json

def export_action_items_to_jira(task_list: list, project_key: str = "SAM") -> dict:
    """
    Mock infrastructure for the Jira Task Exporter Tool (Experiment E5).
    Simulates parsing and sending an array of structured action items to an external project board.
    """
    print(f"📡 Connecting to simulated Jira API for project space: {project_key}...")
    
    created_issues = []
    for idx, task in enumerate(task_list):
        # Build out a mock payload matching a standard ticketing system schema
        issue_mock = {
            "ticket_id": f"{project_key}-{100 + idx}",
            "summary": task.get("task_description", "No description provided"),
            "assignee": task.get("assignee", "Unassigned"),
            "priority": task.get("priority", "Medium"),
            "status": "To Do"
        }
        created_issues.append(issue_mock)
        print(f"✅ Created Jira Ticket {issue_mock['ticket_id']} | Assignee: {issue_mock['assignee']}")
        
    return {
        "status": "success",
        "message": f"Successfully exported {len(created_issues)} tasks to Jira.",
        "payload": created_issues
    }

# --- Strict JSON Schema layout for Member 2's validation layer ---
tool_json_schema = {
    "name": "export_action_items_to_jira",
    "description": "Exports a structured list of action items, assignees, and priorities extracted from a dialogue to a project board.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_key": {"type": "string", "description": "The target project identifier shorthand."},
            "task_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_description": {"type": "string", "description": "The exact action point or deliverable item mentioned."},
                        "assignee": {"type": "string", "description": "The person named in the text responsible for this task."},
                        "priority": {"type": "string", "enum": ["High", "Medium", "Low"], "description": "Urgency tier based on dialogue context."}
                    },
                    "required": ["task_description", "assignee"]
                }
            }
        },
        "required": ["task_list"]
    }
}

if __name__ == "__main__":
    # Local sanity check to verify tool execution outputs
    print("🛠️ Testing tool infrastructure locally...")
    sample_extraction = [
        {"task_description": "Send the revised budget sheet by Friday", "assignee": "Amanda", "priority": "High"},
        {"task_description": "Schedule a follow-up call with the engineering lead", "assignee": "John", "priority": "Low"}
    ]
    response = export_action_items_to_jira(sample_extraction)
    print("\nSample Output Payload:")
    print(json.dumps(response, indent=2))
