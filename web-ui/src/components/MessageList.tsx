import type { ChatMessage } from '../types/messages';
import { ChatMessageComponent } from './ChatMessage';

interface MessageListProps {
  messages: ChatMessage[];
}

export function MessageList({ messages }: MessageListProps) {
  return (
    <div className="space-y-4">
      {messages.map((message) => (
        <ChatMessageComponent key={message.id} message={message} />
      ))}
    </div>
  );
}
