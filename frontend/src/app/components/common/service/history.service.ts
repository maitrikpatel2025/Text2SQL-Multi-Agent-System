import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { config } from '../config';

@Injectable({ providedIn: 'root' })
export class HistoryService {
  private historySubject = new BehaviorSubject<string[]>([]);
  history$: Observable<string[]> = this.historySubject.asObservable();
  constructor(private http: HttpClient) {}
  // private apiUrl = 'http://127.0.0.1:8503/query';
  private apiUrl = `${config.apiUrl}query`;
  sendQuestionToAPI(question: string): Observable<any> {
    return this.http.post<any>(this.apiUrl, {
      user_question: question
    });
  }
  clearHistory() {
  this.historySubject.next([]);
}
  
fetchChatHistory(): void {
  this.http.get<any>(`${config.apiUrl}chat_history`).subscribe({
    next: (response) => {
      if (response.success && response.data) {
        const questions = response.data.map((item: any) => item.User_Question);
        this.historySubject.next(questions);
      }
    },
    error: (err) => {
      console.error('Error fetching chat history', err);
    }
  });
}

  addQuestion(question: string) {
    const current = this.historySubject.value;
    this.historySubject.next([question, ...current]);
  }
}