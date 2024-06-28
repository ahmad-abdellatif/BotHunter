import joblib
import pandas as pd
from github import Github
import statistics
import itertools
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from datetime import timedelta
import argparse
import os

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def time_diff(created_time, updated_time):
    return (updated_time - created_time).total_seconds() / 60.0

def get_cosine_sim(*strs):
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)


def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def compute_diff_cosine_similarity(comments_dict):
    repos_mean = []
    xx = []
    for repo in comments_dict.keys():
        repo_mean = []
        combinations = itertools.combinations(comments_dict[repo], 2)
        for x, y in combinations:
            xx.append(get_cosine_sim(x, y)[0][1])
        if len(repo_mean) >= 1:
            repos_mean.append(statistics.median(repo_mean))
    return xx

def get_repo_users(repo_name,g):
    repo = g.get_repo(repo_name)
    users = []
    try:
        contributers = repo.get_contributors("")
        for con in contributers:
            users.append(con.login)
    except Exception as e:
        print(e)
        return

    return users

def cli():
    repo_events = ["CreateEvent", "DeleteEvent", "ForkEvent", "GollumEvent", "MemberEvent", "PublicEvent", "ReleaseEvent", "SponsorshipEvent", "WatchEvent"]
    pr_events =["PullRequestEvent", "PullRequestReviewCommentEvent"]
    issue_events = ["IssueCommentEvent", "IssuesEvent"]
    commit_events = ["CommitCommentEvent", "PushEvent"]

    columns = ['Total number of Repo activities', 'Unique number of Repo activities', 'Total number of PR activities',
            'Total number of Issue activities', 'Unique number of Issue activities', 'Total number of Commit activities', 'Unique number of Commit activities',
            'Number of following', 'Number of followers', 'Median Creation Time of the first activities',
            'Account tag', 'Account name', 'Account bio', 'Account login', 'Median Activity per Day', 'Cosine similarity', 'Cosine similarity of Comments Before Bot',
            'Cosine similarity of Commit Messages']

    login_names = []
    prediction = {}

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    ## Load the model
    rf = joblib.load(os.path.join(__location__, 'random_forest.joblib'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--key", action='store', type=str, required= True, help="GitHub Token")


    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--u',action='store', type=str, help="User name")
    group.add_argument('--repo',action='store', type=str, help="Repository")
    args = parser.parse_args()



    gh = Github(args.key, per_page=100)
    df = pd.DataFrame(columns=columns)

    if args.u != None:
        login_names.append(args.u)
    else:
        login_names = get_repo_users(args.repo,gh)

    for login_name in login_names:
        repo_subactivies = []
        pr_subactivities = []
        issue_subactivities = []
        commit_subactivities = []
        df_record = []
        try:
            user = gh.get_user(str(login_name))
        except Exception as e:
            print(e)
        events = user.get_events()
        repos = {}

        days = {}
        previous_day = datetime.now()
        isFirstDay = True
        commits_message_dict = {}

        # Check the type of user events
        for event in events:
            day = event.created_at.replace(hour=0, minute=0, second=0, microsecond=0)

            issue_list = []
            type = event.type
            payload = event.payload

            if type in repo_events:
                repo_subactivies.append(type)
            elif type in pr_events:
                pr_subactivities.append(payload['action'])
            elif type in issue_events:
                issue_subactivities.append(payload['action'])

                ## To measure the time and text similarity in the issues
                repo = (payload['issue']["repository_url"]).replace('https://api.github.com/repos/', '')
                if repo in repos.keys():
                    repos[repo].append(payload['issue']['number'])
                else:
                    issue_list.append(payload['issue']['number'])
                    repos[repo] = issue_list

            elif type in commit_events:
                try:
                    if type == "PushEvent":
                        commit_subactivities.append("Push")
                        commits = payload['commits']
                        for commit in commits:
                            #  Check the commits made by "user.name"
                            # (user.login == commit['author']['name']) -> the cases where "user.name" is None
                            if  (user.name == commit['author']['name']) or (user.login == commit['author']['name']) or (user.login == commit['author']['name'].replace("[bot]", "")):
                                message = commit['message']
                                key = event.repo.name
                                if key in commits_message_dict.keys():
                                    commits_message_dict[key].append(message)
                                else:
                                    commits_message_dict[key] = [message]
                    else:
                        commit_subactivities.append("created")
                except Exception as e:
                    print(e)

            # To measure the median number of activities per day
            if isFirstDay:
                previous_day = day
                isFirstDay = False
            elif previous_day != day:
                date_modified = previous_day
                while date_modified > day:
                    date_modified -= timedelta(days=1)
                    days[date_modified] = 0

            if day in days.keys():
                activity = days[day] + 1
                days[day] = activity
            else:
                days[day] = 1
            previous_day = day


        ## The process to measure the time differences
        # https://developer.github.com/v3/issues/issue-event-types/
        sub_events = ["commented", "assigned", "labeled", "locked", "merged", "milestoned", "unlocked", "unlabeled", "closed"]
        first_activities_dict = {}
        last_activities_dict = {}
        comments_dict = {}
        previous_comments_dict = {}
        for key in repos.keys():
            #time.sleep(random.randint(1, 5))
            try:
                repo = gh.get_repo(key)
                repo_issues = list(dict.fromkeys(repos[key]))
                for issue_key in repo_issues:

                    issue = repo.get_issue(issue_key)
                    events_timeline = issue.get_timeline()
                    activity_count = 0
                    is_first_activity = True
                    comment_before_bot_action = None

                    # Itterate through the activities of an issue
                    for comment in events_timeline:

                        activity_count += 1
                        if comment.event in sub_events:

                            # Check if the actor is not a ghost
                            if comment.actor is None:
                                continue

                            if comment.event == "commented" and comment.actor.login == login_name:
                                if key in comments_dict.keys():
                                    comments_dict[key].append(comment.body)
                                else:
                                    comments_dict[key] = [comment.body]

                            # For the text similarity between comments before a bot activity
                            # Check if the current comment is from another account
                            if comment.actor.login != login_name and comment.event == "commented":
                                comment_before_bot_action = comment.body
                            # Check if the current activity is from the account and there is a previous comment from another account
                            elif comment.actor.login == login_name and comment_before_bot_action is not None:
                                if key in previous_comments_dict.keys():
                                    previous_comments_dict[key].append(comment_before_bot_action)
                                else:
                                    previous_comments_dict[key] = [comment_before_bot_action]
                                comment_before_bot_action = None

                            # Check the first activity for the account
                            if comment.actor.login == login_name and is_first_activity:
                                is_first_activity = False

                                if key in first_activities_dict.keys():
                                    first_activities_dict[key].append(time_diff(issue.created_at,comment.created_at))
                                else:
                                    first_activities_dict[key] = [time_diff(issue.created_at,comment.created_at)]

                            #Get Last Event time by the account
                            if comment.actor.login == login_name and activity_count == events_timeline.totalCount:
                                if key in last_activities_dict.keys():
                                    last_activities_dict[key].append(time_diff(previous_comment.created_at,comment.created_at))
                                else:
                                    last_activities_dict[key] = [time_diff(previous_comment.created_at,comment.created_at)]

                            previous_comment = comment
            except Exception as e:
                    print(e)
        user_medians = []
        for repo in first_activities_dict.keys():
            repo_medians = []
            median = statistics.median(first_activities_dict[repo])
            for value in first_activities_dict[repo]:
                repo_medians.append(abs(median-value))
            user_medians.append(statistics.median(repo_medians))

        ## checking the tag assigened to the account
        bot_tag = 0
        if user.type == 'Bot':
            bot_tag = 1

        ## checking the name, bio, and login
        pattern_name = 0
        pattern_bio = 0
        pattern_login = 0
        pattern = re.compile(r'bot$|automate$', re.IGNORECASE)

        if user.name is not None:
            name_match = pattern.search(user.name)
            if name_match:
                pattern_name = 1
        if user.bio is not None:
            bio_match = pattern.search(user.bio)
            if bio_match:
                pattern_bio = 1
        if user.login is not None:
            login_match = pattern.search(user.login)
            if login_match:
                pattern_login = 1

        # Report
        ## Total Number of activities
        df_record.append(len(repo_subactivies))
        ## Total number of unique activities
        df_record.append(len(set(repo_subactivies)))


        df_record.append(len(pr_subactivities))



        df_record.append(len(issue_subactivities))
        ## Total number of unique activities
        df_record.append(len(set(issue_subactivities)))

        df_record.append(len(commit_subactivities))
        ## Total number of unique activities
        df_record.append(len(set(commit_subactivities)))

        df_record.append(user.followers)
        df_record.append(user.following)

        # Median of creation time
        if len(user_medians) > 0:
            df_record.append(statistics.median(user_medians))
        else:
            df_record.append(0)


        df_record.append(bot_tag)


        df_record.append(pattern_name)
        df_record.append(pattern_bio)
        df_record.append(pattern_login)

        if len(days.values()) > 0:
            df_record.append(statistics.median(days.values()))
        else:
            df_record.append(0)

        try:
            bot_comments_cosine_similarity = compute_diff_cosine_similarity(comments_dict)
            if len(bot_comments_cosine_similarity) > 0:
                df_record.append(statistics.median(bot_comments_cosine_similarity))
            else:
                df_record.append(0)
        except Exception as e:
            df_record.append(0)
            print(e)

        try:
            previous_comments_cosine_similarity = compute_diff_cosine_similarity(previous_comments_dict)
            if len(previous_comments_cosine_similarity) > 0:
                df_record.append(statistics.median(previous_comments_cosine_similarity))
            else:
                df_record.append(0)
        except Exception as e:
            df_record.append(0)
            print(e)

        try:
            bot_comments_cosine_similarity = compute_diff_cosine_similarity(commits_message_dict)
            if len(bot_comments_cosine_similarity) > 0:
                df_record.append(statistics.median(bot_comments_cosine_similarity))
            else:
                df_record.append(0)
        except Exception as e:
            df_record.append(0)

        df.loc[len(df)] = df_record


        array = df.values
        X_test = array[:,0:len(df.columns)]

        ## Make the prediction
        res = rf.predict(X_test)

        ## Print the prediction
        if res[0]:
            prediction[login_name] = "Bot"
            # print(login_name + "Bot")
        else:
            prediction[login_name] = "Human"
            # print(login_name + "Human")

    for key in prediction.keys():
        print(key + ":" + prediction[key])

if __name__ == '__main__':
    cli()
