import sys
from collections import defaultdict

def read_events():
    """
    Reads events from stdin.
    Each line is expected to be: event_type,timestamp
    """
    events = []

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            event_type, timestamp = line.split(",")
            events.append({
                "type": event_type,
                "timestamp": timestamp
            })
        except ValueError:
            print(f"Skipping invalid line: {line}")

    return events


def summarize_events(events):
    """
    Counts how many times each event type occurs.
    """
    summary = defaultdict(int)

    for event in events:
        summary[event["type"]] += 1

    return summary


def print_summary(summary):
    """
    Prints event counts.
    """
    print("Event Summary:")
    for event_type, count in summary.items():
        print(f"{event_type}: {count}")


def main():
    events = read_events()

    if not events:
        print("No events found.")
        return

    summary = summarize_events(events)
    print_summary(summary)


if __name__ == "__main__":
    main()